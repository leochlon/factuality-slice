#!/usr/bin/env python3
"""
Factuality Slice Data Pipeline - Public Dataset Curation
Builds evidence-grounded QA dataset from FEVER, HotpotQA, NQ-Open, PopQA

Usage:
  python build_data.py --data-dir data --seed 42 --accept-new-hash
"""

import json
import random
import hashlib
import zipfile
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import requests
import gzip
import shutil
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import re
import math
from urllib.parse import urlparse

# -------------------------
# Source URLs (HTTPS first)
# -------------------------
SOURCES = {
    'fever_train': {
        'url': 'https://fever.ai/download/fever/train.jsonl',
        'fallback_url': None
    },
    'fever_dev': {
        'url': 'https://fever.ai/download/fever/shared_task_dev.jsonl',
        'fallback_url': None
    },
    'fever_wiki': {
        'url': 'https://fever.ai/download/fever/wiki-pages.zip',
        'fallback_url': None
    },
    'hotpot_train': {
        'url': 'https://dl.fbaipublicfiles.com/hotpotqa/hotpot_train_v1.1.json',
        'fallback_url': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json'
    },
    'hotpot_dev': {
        'url': 'https://dl.fbaipublicfiles.com/hotpotqa/hotpot_dev_distractor_v1.json',
        'fallback_url': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json'
    },
    'nq_open': {
        # KILT NQ-Open (dev) used here for convenience; swap as needed
        'url': 'https://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl',
        'fallback_url': None
    },
    'popqa': {
        'url': 'https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv',
        'fallback_url': None
    }
}

# Exact licenses (for datacard)
LICENSES = {
    'fever': 'CC-BY-SA 4.0',
    'hotpot': 'CC-BY-SA 4.0',
    'nq': 'CC-BY 4.0',
    'popqa': 'MIT',
}

# Minimal recency traps (examples)
RECENCY_TRAPS = [
    {
        'question': 'Who is the current president of the United States?',
        'outdated_answer': 'Donald Trump',
        'current_answer': 'Joe Biden',
        'evidence_date': '2019-06-01',
        'evidence': 'Donald Trump is serving as the 45th President of the United States.'
    },
    {
        'question': 'What is the tallest building in the world?',
        'outdated_answer': 'Burj Khalifa',
        'current_answer': 'Burj Khalifa',
        'evidence_date': '2010-01-04',
        'evidence': 'The Burj Khalifa in Dubai was completed in 2010 as the tallest building.'
    }
]

# -------------------------
# Data structures
# -------------------------

@dataclass
class ContextChunk:
    """Enhanced context chunk with provenance"""
    id: str
    text: str
    source: str  # e.g., 'fever_wiki', 'wikipedia_proxy', 'hotpot_context'
    page_title: Optional[str] = None
    sentence_id: Optional[int] = None
    url: Optional[str] = None
    snapshot_date: Optional[str] = None
    retrieval_score: Optional[float] = None

    def to_dict(self):
        d = {
            "id": self.id,
            "text": self.text,
            "source": self.source
        }
        if self.page_title is not None:
            d["page_title"] = self.page_title
        if self.sentence_id is not None:
            d["sentence_id"] = self.sentence_id
        if self.url is not None:
            d["url"] = self.url
        if self.snapshot_date is not None:
            d["snapshot_date"] = self.snapshot_date
        if self.retrieval_score is not None:
            d["retrieval_score"] = self.retrieval_score
        return d

@dataclass
class Sample:
    """Unified sample format with enhanced metadata"""
    id: str
    question: str
    context_chunks: List[Dict[str, Any]]  # List of ContextChunk dicts
    answer: str
    support_spans: List[int]   # indices into context_chunks
    hard_negatives: List[int]  # indices into context_chunks
    metadata: Dict[str, Any]

# -------------------------
# Efficient BM25 Retriever
# -------------------------

class BM25Retriever:
    """
    Inverted-index BM25 retriever.
    - add_documents expects: List[(text, metadata_dict)]
    - search(query, k): returns top-k [(text, metadata_dict, score)]
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Tuple[str, Dict[str, Any]]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # token -> list of (doc_id, tf)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]]):
        self.docs = documents
        self.N = len(documents)
        self.doc_len = [0] * self.N
        # Build postings
        for i, (text, _) in enumerate(documents):
            tokens = self._tokenize(text)
            self.doc_len[i] = len(tokens)
            tf = Counter(tokens)
            for tok, f in tf.items():
                self.postings[tok].append((i, f))
            self.df.update(tf.keys())
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        # IDF
        self.idf = {tok: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0) for tok, df in self.df.items()}

    def search(self, query: str, k: int = 10) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.N == 0:
            return []
        qtokens = self._tokenize(query)
        cand_scores: Dict[int, float] = defaultdict(float)
        for tok in qtokens:
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = self.idf.get(tok, 0.0)
            for doc_id, tf in plist:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                cand_scores[doc_id] += idf * tf * (self.k1 + 1) / denom
        if not cand_scores:
            return []
        top = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.docs[i][0], self.docs[i][1], float(s)) for i, s in top]

    # Persistence helpers
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({
                "k1": self.k1, "b": self.b,
                "docs": self.docs, "doc_len": self.doc_len, "avgdl": self.avgdl, "N": self.N,
                "df": self.df, "idf": self.idf, "postings": self.postings
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        inst = cls(k1=obj["k1"], b=obj["b"])
        inst.docs = obj["docs"]
        inst.doc_len = obj["doc_len"]
        inst.avgdl = obj["avgdl"]
        inst.N = obj["N"]
        inst.df = obj["df"]
        inst.idf = obj["idf"]
        inst.postings = obj["postings"]
        return inst

# -------------------------
# Data Builder
# -------------------------

class DataBuilder:
    def __init__(self, data_dir: Path = Path("data"), seed: int = 42, accept_new_hash: bool = False):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.seed = seed
        self.accept_new_hash = accept_new_hash

        self._fever_wiki: Optional[Dict[Tuple[str, int], str]] = None   # required FEVER wiki only
        self._fever_retriever: Optional[BM25Retriever] = None

        random.seed(seed)

        for d in [self.raw_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Pinned expected hashes (NOT TOFU)
        self.pins_path = self.data_dir / "source_hashes.json"
        self.expected_hashes: Dict[str, str] = {}
        if self.pins_path.exists():
            with open(self.pins_path) as f:
                self.expected_hashes = json.load(f)
        else:
            if not self.accept_new_hash:
                print("   No pinned hash file found (source_hashes.json).")
                print("    Run once with --accept-new-hash to create pins for current downloads.")
            else:
                print("  Will create pinned source hashes (TOFU) for this run.")

        # Paths for persisted FEVER indices
        self.fever_bm25_path = self.raw_dir / "fever_bm25_index.pkl"
        self.fever_wiki_required_path = self.raw_dir / "fever_wiki_required.pkl"

        # Caps & knobs for FEVER indexing and BM25 size
        self.max_bm25_docs = 200_000        # cap number of sentences in BM25
        self.extra_sampling_prob = 0.005    # sampling prob for non-required pages to fill BM25 capacity gradually

    # ---------- Hashing & Downloads ----------

    @staticmethod
    def sha256_file(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    def _update_or_check_pin(self, source_name: str, computed_hash: str) -> bool:
        """If pin exists, verify. If not, require --accept-new-hash to create it."""
        if source_name in self.expected_hashes:
            expected = self.expected_hashes[source_name]
            if computed_hash != expected:
                print(f"   Hash mismatch for {source_name}!")
                print(f"    Expected: {expected[:16]}..., Got: {computed_hash[:16]}...")
                return False
            return True
        else:
            if not self.accept_new_hash:
                print(f"   No pinned hash for {source_name}; refusing TOFU without --accept-new-hash.")
                return False
            # Create pin
            self.expected_hashes[source_name] = computed_hash
            with open(self.pins_path, 'w') as f:
                json.dump(self.expected_hashes, f, indent=2)
            print(f"   Pinned hash for {source_name}: {computed_hash[:16]}...")
            return True

    def download_file(self, source_name: str, timeout: int = 60) -> Optional[Path]:
        """Download with caching and pinned hash verification."""
        source_info = SOURCES.get(source_name, {})
        url = source_info.get('url')
        fallback_url = source_info.get('fallback_url')

        if not url:
            print(f"   No URL configured for {source_name}")
            return None

        filename = source_name + Path(urlparse(url).path).suffix
        filepath = self.raw_dir / filename

        # Cached file: verify pinned hash
        if filepath.exists():
            computed = self.sha256_file(filepath)
            if self._update_or_check_pin(source_name, computed):
                print(f"   Using cached {filename}")
                return filepath
            else:
                print(f"   Removing cached file due to hash mismatch/unpinned: {filename}")
                filepath.unlink(missing_ok=True)

        # Try URLs
        urls_to_try = [url] + ([fallback_url] if fallback_url else [])
        for try_url in urls_to_try:
            host = urlparse(try_url).netloc
            print(f"  ↓ Downloading {filename} from {host}...")
            try:
                resp = requests.get(try_url, stream=True, timeout=timeout)
                resp.raise_for_status()
                temp_path = filepath.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                # Decompress .gz
                if filename.endswith('.gz'):
                    print(f"    Decompressing {filename}...")
                    with gzip.open(temp_path, 'rb') as f_in:
                        with open(temp_path.with_suffix(''), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    temp_path.unlink()
                    temp_path = temp_path.with_suffix('')
                    filepath = filepath.with_suffix('')

                # Move to final
                temp_path.rename(filepath)

                # Verify or pin
                computed = self.sha256_file(filepath)
                if self._update_or_check_pin(source_name, computed):
                    print(f"     Download successful and hash verified/pinned")
                    return filepath
                else:
                    print("     Hash verification failed.")
                    filepath.unlink(missing_ok=True)
                    continue
            except requests.exceptions.RequestException as e:
                print(f"     Failed: {e}")

        print(f"   All download attempts failed for {source_name}")
        return None

    def download_and_extract_zip(self, source_name: str) -> Optional[Path]:
        """Download and extract zip file (no zip-slip protection per user request)."""
        zip_path = self.download_file(source_name)
        if not zip_path:
            return None
        extract_dir = self.raw_dir / source_name
        if extract_dir.exists():
            print(f"     Already extracted to {extract_dir}")
            return extract_dir
        print(f"    Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        return extract_dir

    # ---------- FEVER Helpers (Fix #3) ----------

    def get_required_fever_titles(self) -> Set[str]:
        """Extract all Wikipedia titles referenced by FEVER evidence (train+dev)."""
        required: Set[str] = set()
        for split in ['fever_train', 'fever_dev']:
            path = self.download_file(split)
            if not path:
                continue
            with open(path) as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        for ev_set in item.get('evidence', []):
                            for ev in ev_set:
                                if len(ev) >= 3 and isinstance(ev[2], str):
                                    required.add(ev[2])
                    except Exception:
                        continue
        print(f"    FEVER required titles: {len(required)}")
        return required

    def ensure_fever_indices(self, required_titles: Optional[Set[str]] = None):
        """
        Ensure we have:
        - self._fever_wiki: dict[(title, sent_id)] -> text  (ONLY required titles)
        - self._fever_retriever: BM25 over a capped set of sentences
        Persist both to disk to avoid rebuild.
        """
        # Try to load from disk
        if self._fever_retriever is None and self.fever_bm25_path.exists():
            try:
                self._fever_retriever = BM25Retriever.load(self.fever_bm25_path)
                print("     Loaded FEVER BM25 index from cache.")
            except Exception:
                self._fever_retriever = None

        if self._fever_wiki is None and self.fever_wiki_required_path.exists():
            try:
                with open(self.fever_wiki_required_path, 'rb') as f:
                    self._fever_wiki = pickle.load(f)
                print("     Loaded FEVER required wiki subset from cache.")
            except Exception:
                self._fever_wiki = None

        # If both loaded, return
        if self._fever_retriever is not None and self._fever_wiki is not None:
            return

        # Need to (re)build from wiki zip
        wiki_dir = self.download_and_extract_zip('fever_wiki')
        if not wiki_dir:
            print("     Could not download/extract FEVER wiki")
            return

        if required_titles is None:
            required_titles = self.get_required_fever_titles()

        wiki_files = sorted(wiki_dir.rglob("*.jsonl"))
        required_lookup: Dict[Tuple[str, int], str] = {}
        bm25_docs: List[Tuple[str, Dict[str, Any]]] = []
        bm25_count = 0

        print("    Building FEVER indices (streaming)...")
        for wf in wiki_files:
            # Try multiple encodings to handle potential encoding issues
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(wf, encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"     Skipping {wf.name} - couldn't decode with any encoding")
                continue
            
            for line in lines:
                try:
                    page = json.loads(line)
                    title = page.get('id')
                    if not isinstance(title, str):
                        continue
                    # split lines -> sentences
                    for row in page.get('lines', '').split('\n'):
                        if not row.strip():
                            continue
                        parts = row.split('\t', 1)
                        if len(parts) != 2 or not parts[0].isdigit():
                            continue
                        sent_id = int(parts[0])
                        text = parts[1]
                        # If this title is required, store into wiki index
                        if title in required_titles:
                            required_lookup[(title, sent_id)] = text
                            # Also add to BM25
                            if bm25_count < self.max_bm25_docs:
                                bm25_docs.append((text, {'title': title, 'sent_id': sent_id}))
                                bm25_count += 1
                        else:
                            # Sample extra pages into BM25 to broaden coverage
                            if bm25_count < self.max_bm25_docs and random.random() < self.extra_sampling_prob:
                                bm25_docs.append((text, {'title': title, 'sent_id': sent_id}))
                                bm25_count += 1
                except Exception:
                    continue

        # Persist required wiki subset
        self._fever_wiki = required_lookup
        with open(self.fever_wiki_required_path, 'wb') as f:
            pickle.dump(self._fever_wiki, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"     FEVER required wiki subset: {len(self._fever_wiki)} sentences")

        # Build and persist BM25
        self._fever_retriever = BM25Retriever()
        self._fever_retriever.add_documents(bm25_docs)
        self._fever_retriever.save(self.fever_bm25_path)
        print(f"     FEVER BM25 built with {len(bm25_docs)} sentences and saved")

        
    # ---------- Dataset processors ----------

    def cap_with_all_support(self,
                             context_chunks: List[ContextChunk],
                             support_spans: List[int],
                             hard_negatives: List[int],
                             max_chunks: int = 10,
                             max_distractors: int = 3) -> Tuple[List[Dict], List[int], List[int]]:
        """Ensure all supporting evidence is kept, then add distractors up to cap."""
        keep = set(support_spans)
        dcount = 0
        for idx in hard_negatives:
            if len(keep) >= max_chunks:
                break
            if idx not in keep and dcount < max_distractors:
                keep.add(idx)
                dcount += 1
        keep = sorted(keep)
        remap = {old: new for new, old in enumerate(keep)}
        new_chunks = [context_chunks[i].to_dict() if isinstance(context_chunks[i], ContextChunk) else context_chunks[i] for i in keep]
        new_support = [remap[i] for i in support_spans if i in remap]
        new_negs = [remap[i] for i in hard_negatives if i in remap and remap[i] not in new_support]
        return new_chunks, new_support, new_negs

    def process_fever(self, split: str = 'train', max_samples: int = 2000) -> List[Sample]:
        """Process FEVER with proper evidence coverage (uses required wiki subset & BM25)."""
        samples: List[Sample] = []
        path = self.download_file(f'fever_{split}')
        if not path:
            print(f"   Could not download FEVER {split}, skipping")
            return samples

        print("    Preparing FEVER indices...")
        required_titles = self.get_required_fever_titles()
        self.ensure_fever_indices(required_titles)

        if not self._fever_wiki:
            print("   FEVER wiki subset unavailable, skipping FEVER.")
            return samples

        processed_count = 0
        skipped_no_evidence = 0

        with open(path) as f:
            for line_no, line in enumerate(f):
                if line_no >= max_samples:
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if item.get('label') == 'NOT ENOUGH INFO':
                    continue  # controls added elsewhere

                question = f"Is the following claim true or false? {item['claim']}"
                answer = "True" if item['label'] == 'SUPPORTS' else "False"

                context_chunks: List[ContextChunk] = []
                support_spans: List[int] = []
                hard_negatives: List[int] = []

                # Add gold evidence (from required wiki subset)
                for ev_set in item.get('evidence', []):
                    if not ev_set:
                        continue
                    for ev in ev_set[:3]:
                        if len(ev) >= 4:
                            title, line_no_ev = ev[2], ev[3]
                            text = self._fever_wiki.get((title, line_no_ev))
                            if text:
                                chunk = ContextChunk(
                                    id=f"fever_{title}_{line_no_ev}",
                                    text=text,
                                    source='fever_wiki',
                                    page_title=title,
                                    sentence_id=line_no_ev,
                                    snapshot_date='2017-06-01'
                                )
                                context_chunks.append(chunk)
                                support_spans.append(len(context_chunks) - 1)

                if not context_chunks:
                    skipped_no_evidence += 1
                    continue

                # Add BM25 hard negatives (topical)
                if self._fever_retriever:
                    claim = item['claim']
                    retrieved = self._fever_retriever.search(claim, k=20)
                    gold_keys = {(ev[2], ev[3]) for ev_set in item.get('evidence', []) for ev in ev_set if len(ev) >= 4}
                    for ret_text, ret_meta, score in retrieved:
                        key = (ret_meta['title'], ret_meta['sent_id'])
                        if key in gold_keys:
                            continue
                        chunk = ContextChunk(
                            id=f"fever_d_{ret_meta['title']}_{ret_meta['sent_id']}",
                            text=ret_text,
                            source='fever_wiki',
                            page_title=ret_meta['title'],
                            sentence_id=ret_meta['sent_id'],
                            retrieval_score=float(score),
                            snapshot_date='2017-06-01'
                        )
                        context_chunks.append(chunk)
                        hard_negatives.append(len(context_chunks) - 1)
                        if len(hard_negatives) >= 5:
                            break

                final_chunks, final_support, final_negatives = self.cap_with_all_support(
                    context_chunks, support_spans, hard_negatives, max_chunks=10
                )

                samples.append(Sample(
                    id=f"fever_{item['id']}",
                    question=question,
                    context_chunks=final_chunks,
                    answer=answer,
                    support_spans=final_support,
                    hard_negatives=final_negatives,
                    metadata={
                        "source": "fever",
                        "label": item['label'],
                        "evidence_coverage": len(final_support) / max(1, len(support_spans))
                    }
                ))
                processed_count += 1

        print(f"    Processed {processed_count} FEVER samples, skipped {skipped_no_evidence} (no evidence)")
        return samples

    def process_hotpot(self, split: str = 'train', max_samples: int = 2000) -> List[Sample]:
        """Process HotpotQA preserving multi-hop integrity."""
        samples: List[Sample] = []
        path = self.download_file(f'hotpot_{split}')
        if not path:
            print(f"   Could not download HotpotQA {split}, skipping")
            return samples

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"   Error reading HotpotQA {split}: {e}")
            return samples

        multi_hop_preserved = 0

        for item in data[:max_samples]:
            context_chunks: List[ContextChunk] = []
            support_spans: List[int] = []
            hard_negatives: List[int] = []
            support_set = {(t, sid) for t, sid in item.get('supporting_facts', [])}

            for title, sentences in item.get('context', []):
                for sent_id, sent_text in enumerate(sentences):
                    chunk = ContextChunk(
                        id=f"hotpot_{title.replace(' ', '_')}_{sent_id}",
                        text=sent_text,
                        source='hotpot_context',
                        page_title=title,
                        sentence_id=sent_id,
                        snapshot_date='2017-10-01'
                    )
                    context_chunks.append(chunk)
                    if (title, sent_id) in support_set:
                        support_spans.append(len(context_chunks) - 1)
                    else:
                        hard_negatives.append(len(context_chunks) - 1)

            if len(support_spans) >= 2:
                multi_hop_preserved += 1

            final_chunks, final_support, final_negatives = self.cap_with_all_support(
                context_chunks, support_spans, hard_negatives,
                max_chunks=12, max_distractors=4
            )

            samples.append(Sample(
                id=f"hotpot_{hashlib.md5(item['question'].encode()).hexdigest()[:8]}",
                question=item['question'],
                context_chunks=final_chunks,
                answer=item['answer'],
                support_spans=final_support,
                hard_negatives=final_negatives,
                metadata={
                    "source": "hotpot",
                    "type": item.get('type'),
                    "level": item.get('level'),
                    "multi_hop": len(final_support) >= 2,
                    "evidence_coverage": len(final_support) / max(1, len(support_spans))
                }
            ))

        print(f"    Multi-hop preserved: {multi_hop_preserved}/{len(samples)}")
        return samples

    def process_nq_open(self, max_samples: int = 1000) -> List[Sample]:
        """Process NQ-Open (KILT dev) with retrieval using FEVER BM25 as a proxy."""
        samples: List[Sample] = []
        path = self.download_file('nq_open')
        if not path:
            print(f"   Could not download NQ-Open, skipping")
            return samples

        # Ensure FEVER BM25 exists
        self.ensure_fever_indices()
        if not self._fever_retriever:
            print("   No retriever available for NQ-Open, skipping")
            return samples

        processed = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                question = item.get('input', '')
                outputs = item.get('output', [])
                if not question or not outputs:
                    continue

                answer = outputs[0].get('answer', '')
                if not answer:
                    continue

                query = f"{question} {answer}"
                retrieved = self._fever_retriever.search(query, k=10)

                context_chunks: List[ContextChunk] = []
                support_spans: List[int] = []
                hard_negatives: List[int] = []

                for ret_text, ret_meta, score in retrieved:
                    chunk = ContextChunk(
                        id=f"nq_{ret_meta['title']}_{ret_meta['sent_id']}",
                        text=ret_text,
                        source='wikipedia_proxy',
                        page_title=ret_meta['title'],
                        sentence_id=ret_meta['sent_id'],
                        retrieval_score=float(score),
                        snapshot_date='2017-06-01'
                    )
                    context_chunks.append(chunk)
                    if answer.lower() in ret_text.lower():
                        support_spans.append(len(context_chunks) - 1)
                    else:
                        hard_negatives.append(len(context_chunks) - 1)

                if not context_chunks:
                    continue

                if not support_spans:
                    support_spans = [0]
                    hard_negatives = list(range(1, min(5, len(context_chunks))))

                final_chunks, final_support, final_negatives = self.cap_with_all_support(
                    context_chunks, support_spans, hard_negatives, max_chunks=8
                )

                samples.append(Sample(
                    id=f"nq_{item.get('id', str(i))}",
                    question=question,
                    context_chunks=final_chunks,
                    answer=answer[:100],
                    support_spans=final_support,
                    hard_negatives=final_negatives,
                    metadata={
                        "source": "nq_open",
                        "proxy_wikipedia": True
                    }
                ))
                processed += 1

        print(f"    Processed {processed} NQ-Open samples")
        return samples

    def process_popqa(self, max_samples: int = 500) -> List[Sample]:
        """Process PopQA (long-tail) with retrieval via FEVER BM25 as proxy."""
        samples: List[Sample] = []
        path = self.download_file('popqa')
        if not path:
            print(f"   Could not download PopQA, skipping")
            return samples

        self.ensure_fever_indices()
        if not self._fever_retriever:
            print("   No retriever available for PopQA, skipping")
            return samples

        processed = 0
        weak_count = 0

        with open(path, 'r', encoding='utf-8') as f:
            # Skip header if present
            first_line = f.readline()
            if 'question' in first_line.lower() and 'answer' in first_line.lower():
                pass  # Header line, continue
            else:
                f.seek(0)  # No header, reset
            
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                question = parts[0].strip()
                answer = parts[1].strip()
                
                # Get popularity score if available (3rd column)
                s_pop = -1
                if len(parts) >= 3:
                    try:
                        s_pop = float(parts[2])
                    except ValueError:
                        pass
                
                # Filter for long-tail
                if s_pop > 50:
                    continue
                
                if not question or not answer:
                    continue

                query = f"{question} {answer}"
                retrieved = self._fever_retriever.search(query, k=10)

                context_chunks: List[ContextChunk] = []
                support_spans: List[int] = []
                hard_negatives: List[int] = []
                found_evidence = False

                for ret_text, ret_meta, score in retrieved:
                    chunk = ContextChunk(
                        id=f"popqa_{ret_meta['title']}_{ret_meta['sent_id']}",
                        text=ret_text,
                        source='wikipedia_proxy',
                        page_title=ret_meta['title'],
                        sentence_id=ret_meta['sent_id'],
                        retrieval_score=float(score),
                        snapshot_date='2017-06-01'
                    )
                    context_chunks.append(chunk)
                    if answer.lower() in ret_text.lower():
                        support_spans.append(len(context_chunks) - 1)
                        found_evidence = True
                    else:
                        hard_negatives.append(len(context_chunks) - 1)

                if not context_chunks:
                    continue

                if not found_evidence:
                    support_spans = [0]
                    weak_count += 1

                final_chunks, final_support, final_negatives = self.cap_with_all_support(
                    context_chunks, support_spans, hard_negatives, max_chunks=6
                )

                samples.append(Sample(
                    id=f"popqa_{i}",
                    question=question,
                    context_chunks=final_chunks,
                    answer=answer,
                    support_spans=final_support,
                    hard_negatives=final_negatives,
                    metadata={
                        "source": "popqa",
                        "popularity_score": s_pop,
                        "proxy_wikipedia": True,
                        "weak_evidence": not found_evidence
                    }
                ))
                processed += 1

        print(f"    Processed {processed} PopQA samples")
        if weak_count > 0:
            print(f"     {weak_count} PopQA samples with weak evidence (top-1 used)")
        return samples

    def add_control_examples(self, samples: List[Sample]) -> List[Sample]:
        """Add control examples: FEVER NEI + recency traps."""
        control_samples: List[Sample] = []

        fever_train_path = self.raw_dir / 'fever_train.jsonl'
        if fever_train_path.exists():
            nei_count = 0
            self.ensure_fever_indices()
            with open(fever_train_path) as f:
                for line in f:
                    if nei_count >= 300:
                        break
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if item.get('label') != 'NOT ENOUGH INFO':
                        continue

                    question = f"Is the following claim true or false? {item['claim']}"
                    context_chunks: List[ContextChunk] = []
                    if self._fever_retriever:
                        retrieved = self._fever_retriever.search(item['claim'], k=5)
                        for ret_text, ret_meta, score in retrieved:
                            context_chunks.append(ContextChunk(
                                id=f"control_{ret_meta['title']}_{ret_meta['sent_id']}",
                                text=ret_text,
                                source='fever_wiki',
                                page_title=ret_meta['title'],
                                sentence_id=ret_meta['sent_id'],
                                retrieval_score=float(score),
                                snapshot_date='2017-06-01'
                            ))
                    if context_chunks:
                        control_samples.append(Sample(
                            id=f"control_nei_{item['id']}",
                            question=question,
                            context_chunks=[c.to_dict() for c in context_chunks],
                            answer="[INSUFFICIENT_EVIDENCE]",
                            support_spans=[],
                            hard_negatives=list(range(len(context_chunks))),
                            metadata={
                                "source": "control",
                                "control_type": "insufficient_evidence",
                                "origin": "fever_nei"
                            }
                        ))
                        nei_count += 1

        # Recency traps
        for i, trap in enumerate(RECENCY_TRAPS):
            chunk = ContextChunk(
                id=f"recency_{i}",
                text=trap['evidence'],
                source='synthetic',
                snapshot_date=trap['evidence_date']
            )
            control_samples.append(Sample(
                id=f"control_recency_{i}",
                question=trap['question'],
                context_chunks=[chunk.to_dict()],
                answer="[OUTDATED_EVIDENCE]",
                support_spans=[],
                hard_negatives=[0],
                metadata={
                    "source": "control",
                    "control_type": "recency_trap",
                    "evidence_date": trap['evidence_date'],
                    "outdated_answer": trap['outdated_answer'],
                    "current_answer": trap['current_answer']
                }
            ))

        print(f"    Added {len(control_samples)} control samples")
        return samples + control_samples

    # ---------- Validation, Split, Save, Datacard ----------

    @staticmethod
    def validate_and_clean(sample: Sample) -> Optional[Sample]:
        n = len(sample.context_chunks)
        for idx in sample.support_spans:
            if not (0 <= idx < n):
                return None
        for idx in sample.hard_negatives:
            if not (0 <= idx < n):
                return None
        # Remove any overlap
        sset = set(sample.support_spans)
        sample.hard_negatives = [i for i in sample.hard_negatives if i not in sset]
        return sample

    def split_data(self, samples: List[Sample], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Sample]]:
        valid: List[Sample] = []
        for s in samples:
            s2 = self.validate_and_clean(s)
            if s2:
                valid.append(s2)

        # Deduplicate by (question, answer) to reduce leakage across splits
        seen = set()
        deduped: List[Sample] = []
        for s in valid:
            key = (s.question.strip().lower(), s.answer.strip().lower())
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        by_source: Dict[str, List[Sample]] = defaultdict(list)
        for s in deduped:
            src = s.metadata.get('source', 'unknown')
            by_source[src].append(s)

        splits = {'train': [], 'val': [], 'test': []}
        rng = random.Random(self.seed)

        for src, items in sorted(by_source.items()):
            rng.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            splits['train'].extend(items[:n_train])
            splits['val'].extend(items[n_train:n_train+n_val])
            splits['test'].extend(items[n_train+n_val:])

        for k in splits:
            rng.shuffle(splits[k])

        return splits

    @staticmethod
    def save_jsonl(samples: List[Sample], filepath: Path):
        with open(filepath, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + '\n')

    @staticmethod
    def compute_statistics(samples: List[Sample]) -> Dict[str, Any]:
        stats = {
            'total': len(samples),
            'by_source': Counter(),
            'by_control_type': Counter(),
            'answer_types': Counter(),
            'chunk_lengths': [],
            'support_counts': [],
            'negative_counts': [],
            'multi_hop': 0,
            'evidence_coverage': [],
            'answer_in_context': 0
        }
        for s in samples:
            src = s.metadata.get('source', 'unknown')
            stats['by_source'][src] += 1
            if src == 'control':
                stats['by_control_type'][s.metadata.get('control_type', 'unknown')] += 1

            if s.answer in ('[INSUFFICIENT_EVIDENCE]', '[OUTDATED_EVIDENCE]'):
                stats['answer_types'][s.answer] += 1
            elif s.answer in ('True', 'False'):
                stats['answer_types']['boolean'] += 1
            else:
                stats['answer_types']['span'] += 1

            stats['chunk_lengths'].append(len(s.context_chunks))
            stats['support_counts'].append(len(s.support_spans))
            stats['negative_counts'].append(len(s.hard_negatives))
            if len(s.support_spans) >= 2:
                stats['multi_hop'] += 1

            if 'evidence_coverage' in s.metadata:
                stats['evidence_coverage'].append(s.metadata['evidence_coverage'])

            ctx_text = ' '.join([c['text'] for c in s.context_chunks])
            if s.answer and s.answer not in ('[INSUFFICIENT_EVIDENCE]', '[OUTDATED_EVIDENCE]') and s.answer.lower() in ctx_text.lower():
                stats['answer_in_context'] += 1
        return stats

    def generate_datacard(self, splits: Dict[str, List[Sample]]):
        all_stats = {name: self.compute_statistics(samps) for name, samps in splits.items()}

        datacard = f"""# Factuality Slice Dataset Card

## Overview
Evidence-grounded QA dataset for factuality-focused RLHF/RLAIF training.
Generated: {datetime.now().isoformat()}
Seed: {self.seed}
Version: 2.1.0

## Data Integrity
Source files verified against **pinned SHA-256 hashes** (source_hashes.json).
First run requires `--accept-new-hash` to create pins; subsequent runs verify and fail on mismatch.

## Sources & Licenses
- **FEVER**: Fact verification (claims → true/false) - {LICENSES['fever']}
- **HotpotQA**: Multi-hop reasoning - {LICENSES['hotpot']}
- **NQ-Open**: Open-domain QA (KILT) - {LICENSES['nq']}
- **PopQA**: Long-tail entity questions - {LICENSES['popqa']}
- **Controls**: FEVER NEI + Recency traps

## Statistics by Split
| Split | Total | FEVER | Hotpot | NQ-Open | PopQA | Controls |
|------:|------:|------:|-------:|--------:|------:|---------:|
"""
        for split in ['train', 'val', 'test']:
            s = all_stats.get(split, {})
            if not s:
                continue
            by = s['by_source']
            datacard += f"| {split} | {s['total']} | {by.get('fever', 0)} | {by.get('hotpot', 0)} | {by.get('nq_open', 0)} | {by.get('popqa', 0)} | {by.get('control', 0)} |\n"

        datacard += "\n## Evidence Quality Metrics\n"
        datacard += "| Split | Avg Chunks | Avg Support | Multi-hop % | Answer-in-Context % |\n"
        datacard += "|------:|-----------:|------------:|------------:|--------------------:|\n"
        for split in ['train', 'val', 'test']:
            s = all_stats.get(split, {})
            if not s:
                continue
            avg_chunks = (sum(s['chunk_lengths']) / len(s['chunk_lengths'])) if s['chunk_lengths'] else 0
            avg_support = (sum(s['support_counts']) / len(s['support_counts'])) if s['support_counts'] else 0
            multi_hop_pct = (100 * s['multi_hop'] / s['total']) if s['total'] else 0
            leakage_pct = (100 * s['answer_in_context'] / s['total']) if s['total'] else 0
            datacard += f"| {split} | {avg_chunks:.1f} | {avg_support:.1f} | {multi_hop_pct:.1f}% | {leakage_pct:.1f}% |\n"

        datacard += f"""

## Format
```json
{{
  "id": "unique_identifier",
  "question": "Question text",
  "context_chunks": [
    {{
      "id": "chunk_id",
      "text": "evidence text",
      "source": "wikipedia|fever_wiki|hotpot_context",
      "page_title": "...",
      "sentence_id": 12,
      "snapshot_date": "YYYY-MM-DD",
      "retrieval_score": 8.5
    }},
    ...
  ],
  "answer": "Answer text or [INSUFFICIENT_EVIDENCE] or [OUTDATED_EVIDENCE]",
  "support_spans": [0, 2],
  "hard_negatives": [1, 3],
  "metadata": {{
    "source": "fever|hotpot|nq_open|popqa|control",
    "evidence_coverage": 1.0,
    "multi_hop": true
  }}
}}

Key Implementation Notes (v2.1.0)

Inverted-index BM25 with postings (fast).

Pinned source hashes (no trust-on-first-use by default).

FEVER wiki streaming index: required titles only + sampled extras for BM25; persisted to disk.

"""
        with open(self.data_dir / "datacard.md", 'w', encoding='utf-8') as f:
            f.write(datacard)
        print("  Generated comprehensive datacard.md")

# ---------- Build pipeline ----------

    def build(self):
        print("=== Factuality Slice Data Pipeline v2.1.0 ===")
        print(f"Seed: {self.seed}")
        print(f"Date: {datetime.now().isoformat()}\n")

        all_samples: List[Sample] = []

        print("1. Processing FEVER...")
        fever_samples = self.process_fever('train', max_samples=2000)
        all_samples.extend(fever_samples)
        print(f"   Added {len(fever_samples)} FEVER samples")

        print("\n2. Processing HotpotQA...")
        hotpot_samples = self.process_hotpot('train', max_samples=2000)
        all_samples.extend(hotpot_samples)
        print(f"   Added {len(hotpot_samples)} HotpotQA samples")

        print("\n3. Processing NQ-Open (KILT)...")
        nq_samples = self.process_nq_open(max_samples=1000)
        all_samples.extend(nq_samples)
        print(f"   Added {len(nq_samples)} NQ-Open samples")

        print("\n4. Processing PopQA (long-tail)...")
        popqa_samples = self.process_popqa(max_samples=500)
        all_samples.extend(popqa_samples)
        print(f"   Added {len(popqa_samples)} PopQA samples")

        if not all_samples:
            print("\n Error: No samples collected from datasets.")
            return {}

        print("\n5. Adding control examples...")
        all_samples = self.add_control_examples(all_samples)

        print("\n6. Splitting data with stratification and deduplication...")
        splits = self.split_data(all_samples)

        print("\n7. Saving processed data...")
        for split_name, split_samples in splits.items():
            filepath = self.processed_dir / f"{split_name}.jsonl"
            self.save_jsonl(split_samples, filepath)
            print(f"   {split_name}: {len(split_samples)} samples → {filepath}")

        self.generate_datacard(splits)

        # Save (or update) pinned hashes if we accepted new hashes
        if self.accept_new_hash:
            with open(self.pins_path, 'w') as f:
                json.dump(self.expected_hashes, f, indent=2)
            print("   Saved source_hashes.json (pinned hashes)")

        print("\n✅ Data pipeline complete!")
        print(f"   Total samples: {len(all_samples)}")
        print(f"   Output dir: {self.processed_dir}")

        return splits


def main():
    import argparse
    parser = argparse.ArgumentParser(
    description="Factuality Slice Data Pipeline v2.1.0 - Build evidence-grounded QA dataset"
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
    help='Data directory (default: data/)')
    parser.add_argument('--seed', type=int, default=42,
    help='Random seed (default: 42)')
    parser.add_argument('--accept-new-hash', action='store_true',
    help='Allow creating pinned hashes on first run (required if source_hashes.json not present)')
    args = parser.parse_args()

    builder = DataBuilder(args.data_dir, seed=args.seed, accept_new_hash=args.accept_new_hash)
    splits = builder.build()
    return 0 if splits and all(len(s) > 0 for s in splits.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())