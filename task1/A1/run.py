#!/usr/bin/env python3

import argparse
import re
import math
from collections import Counter
import sys, os
import xml.etree.ElementTree as ET
import simplemma

parser = argparse.ArgumentParser(description="Assignment 1")

parser.add_argument("-q", "--query", type=str, default="topics.xml" ,help="A query string")
parser.add_argument("-d", "--documents", type=str, default="documents.lst", help="Name of documents file")
parser.add_argument("-r", "--run", type=str, default="run", help="Run prefix name")
parser.add_argument("-o", "--output", type=str, default="out", help="Output file")
parser.add_argument("-l", "--language", type=str, default="cs", help="Language (en or cs)"  )
parser.add_argument("-p", "--prefix", type=str, default=None, help="All files prefix")
parser.add_argument("--lowercase", default=False, action="store_true", help="Lowercase all tokens")
parser.add_argument("--stopwords", type=float, default=0.0, help="Remove top percentage of stopwords (0.0-1.0)")
parser.add_argument("--stopwords_probabs", default=False, action="store_true", help="Remove stopwords based on probability instead of frequency")
parser.add_argument("--lemmatize", default=False, action="store_true", help="Use Morphodita lemmatization")
parser.add_argument("--tfidf", action="store_true", help="Use TF-IDF weighting")

OUTPUT_FOLDER = "outputs/"
DOCUMENTS_FOLDER_CS = "documents_cs/"
DOCUMENTS_FOLDER_EN = "documents_en/"

# TODO add arugments for nlp methods

# TOKENIZATION

# def tokenize(text):
#     return [t for t in re.split(r"[^A-Za-z0-9]+", text) if t]
def tokenize(text, lowercase=False):
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens

def compute_global_frequencies(docs):
    freq = Counter()
    for doc in docs.values():
        freq.update(doc)
    return freq

def get_stopwords2(freq, percentage):
    if percentage <= 0:
        return set()

    sorted_terms = freq.most_common()
    total_tokens = sum(freq.values())

    print(f"Total unique terms: {len(sorted_terms)}")
    print(f"Total term occurrences: {total_tokens}")

    stopwords = set()

    cumulative = 0
    cutoff_tokens = total_tokens * percentage
    
    print(f"Cutoff (top {percentage*100}% token mass): {cutoff_tokens}")

    for term, count in sorted_terms:
        cumulative += count
        stopwords.add(term)

        if cumulative >= cutoff_tokens:
            break

    print(f"Selected {len(stopwords)} stopwords covering {cumulative/total_tokens:.4f} of corpus")

    # print("\nTop stopwords:")
    # for term in list(stopwords)[:20]:
    #     print(term)
    print("Stop words \n", list(stopwords))

    return stopwords

def get_stopwords(freq, percentage):
    if percentage <= 0:
        return set()

    sorted_terms = freq.most_common()
    cutoff = int(len(sorted_terms) * percentage)

    print(f"Total unique terms: {len(sorted_terms)}")
    print(f"Total term occurrences: {sum(freq.values())}")
    print(f"Cutoff for stopwords (top {percentage*100}%): {cutoff} terms")
    print(" ")

    # print the top stopwords
    print(f"Top {cutoff} stopwords:")
    for term, count in sorted_terms[:cutoff]:
        print(f"{term}: {count}")

    return set(term for term, _ in sorted_terms[:cutoff])

def remove_stopwords_from_docs(docs, stopwords):
    new_docs = {}
    for docno, vec in docs.items():
        new_vec = Counter({t: c for t, c in vec.items() if t not in stopwords})
        new_docs[docno] = new_vec
    return new_docs

def remove_stopwords_from_queries(queries, stopwords):
    new_queries = {}
    for qid, vec in queries.items():
        new_vec = Counter({t: c for t, c in vec.items() if t not in stopwords})
        new_queries[qid] = new_vec
    return new_queries

def parse_documents(doc_list_file, language="cs"):
    docs = {}
    print("Parsing documents...", end=" ")

    with open(doc_list_file) as f:
        for i,path in enumerate(f):
            path = path.strip()
            path = DOCUMENTS_FOLDER_CS + path if language == "cs" else DOCUMENTS_FOLDER_EN + path
            if not path:
                continue

            with open(path, encoding="utf-8", errors="ignore") as file:
                content = file.read()

            for doc in re.findall(r"<DOC>(.*?)</DOC>", content, re.DOTALL):
                docno_match = re.search(r"<DOCNO>(.*?)</DOCNO>", doc)
                if not docno_match:
                    continue

                docno = docno_match.group(1).strip()
                
                # IF we want only titles
                # title_match = re.search(r"<TITLE>(.*?)</TITLE>", doc, re.DOTALL)
                # if not title_match:
                    # continue  # skip docs without title
                # title_text = title_match.group(1)

                # remove tags
                text = re.sub(r"<.*?>", " ", doc)

                tokens = tokenize(text, lowercase=main_args.lowercase)
                docs[docno] = Counter(tokens)
            
            # print(f"Parsed {len(docs)} documents. from {i+1} files")
    print("Done.")
    return docs

def parse_queries(xml_file):
    print("Parsing queries...", end=" ")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    queries = {}

    for top in root.findall(".//top"):
        qid = top.find("num").text.strip()
        title = top.find("title").text.strip()
        tokens = tokenize(title, lowercase=main_args.lowercase)
        queries[qid] = Counter(tokens)

    print("Done.")
    return queries

def build_lemma_mapping(vocab, language):
    lemma_map = {}

    # simplemma expects ISO codes like "en", "cs"
    lang = language

    for word in vocab:
        lemma = simplemma.lemmatize(word, lang=lang)

        # fallback safety
        if not lemma:
            lemma = word

        lemma_map[word] = lemma

    return lemma_map

def apply_lemma_mapping_docs(docs, lemma_map):
    new_docs = {}
    for docno, vec in docs.items():
        new_vec = Counter()
        for word, count in vec.items():
            lemma = lemma_map.get(word, word)
            new_vec[lemma] += count
        new_docs[docno] = new_vec
    return new_docs

def apply_lemma_mapping_queries(queries, lemma_map):
    new_queries = {}
    for qid, vec in queries.items():
        new_vec = Counter()
        for word, count in vec.items():
            lemma = lemma_map.get(word, word)
            new_vec[lemma] += count
        new_queries[qid] = new_vec
    return new_queries


def compute_df(docs):
    df = Counter()
    for vec in docs.values():
        for term in vec:
            df[term] += 1
    return df

def build_tfidf(docs, df, N):
    tfidf_docs = {}

    for docno, vec in docs.items():
        new_vec = {}

        for term, tf in vec.items():
            if tf <= 0:
                continue

            tf_weight = 1 + math.log(tf)
            idf = math.log(N / df[term])

            new_vec[term] = tf_weight * idf

        tfidf_docs[docno] = new_vec

    return tfidf_docs


def build_tfidf_queries(queries, df, N):
    tfidf_queries = {}

    for qid, vec in queries.items():
        new_vec = {}

        for term, tf in vec.items():
            if tf <= 0:
                continue

            tf_weight = 1 + math.log(tf)

            # important: handle unseen terms
            if term in df:
                idf = math.log(N / df[term])
            else:
                idf = 0  # or skip

            new_vec[term] = tf_weight * idf

        tfidf_queries[qid] = new_vec

    return tfidf_queries


def cosine(q, d):
    dot = sum(q[t] * d.get(t, 0) for t in q)

    if dot == 0:
        return 0.0

    q_norm = math.sqrt(sum(v * v for v in q.values()))
    d_norm = math.sqrt(sum(v * v for v in d.values()))

    if q_norm == 0 or d_norm == 0:
        return 0.0

    return dot / (q_norm * d_norm)


# MAIN

def main(args):
    docs = parse_documents(args.documents, args.language)
    queries = parse_queries(args.query)
    
    if args.lemmatize:
        print("Lemmatizing with simplemma...")

        vocab = set()
        for d in docs.values():
            vocab.update(d.keys())
        for q in queries.values():
            vocab.update(q.keys())

        print(f"Vocabulary size before lemmatization: {len(vocab)}")

        lemma_map = build_lemma_mapping(vocab, args.language)

        docs = apply_lemma_mapping_docs(docs, lemma_map)
        queries = apply_lemma_mapping_queries(queries, lemma_map)

        print(f"Vocabulary size after lemmatization: {len(set(lemma_map.values()))}")

    if args.stopwords > 0:
        print("Computing global term frequencies and determining stopwords...")
        freq = compute_global_frequencies(docs)
        stopwords = get_stopwords2(freq, args.stopwords)
        
        print("Removing stop words from documents and queries...")
        docs = remove_stopwords_from_docs(docs, stopwords)
        queries = remove_stopwords_from_queries(queries, stopwords)

    if args.tfidf:
        print("Computing TF-IDF...")

        N = len(docs)
        df = compute_df(docs)

        docs = build_tfidf(docs, df, N)
        queries = build_tfidf_queries(queries, df, N)

    with open(OUTPUT_FOLDER + args.output, "w") as out:
        for qid, q_vec in queries.items():
            scores = []

            for docno, d_vec in docs.items():
                s = cosine(q_vec, d_vec)
                if s > 0:
                    scores.append((docno, s))

            scores.sort(key=lambda x: x[1], reverse=True)

            for rank, (docno, score) in enumerate(scores[:1000]):
                out.write(f"{qid} 0 {docno} {rank} {score} {args.run}\n")


if __name__ == "__main__":
    # how to print the current program path? 
    print(f"Current program path: {os.path.abspath(__file__)}")
    print(f"Program arguments: {sys.argv[1:]}")
    
    main_args = parser.parse_args()
    
    OUTPUT_FOLDER = main_args.prefix + OUTPUT_FOLDER if main_args.prefix else OUTPUT_FOLDER
    DOCUMENTS_FOLDER_CS = main_args.prefix + DOCUMENTS_FOLDER_CS if main_args.prefix else DOCUMENTS_FOLDER_CS
    DOCUMENTS_FOLDER_EN = main_args.prefix + DOCUMENTS_FOLDER_EN if main_args.prefix else DOCUMENTS_FOLDER_EN
    main_args.query = main_args.prefix + main_args.query if main_args.prefix else main_args.query
    main_args.documents = main_args.prefix + main_args.documents if main_args.prefix else main_args.documents
    
    if main_args.stopwords_probabs:
        output_basename = main_args.output.replace(".res", "")
        for stopword_percentage in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            print(f"Running with stopword removal percentage: {stopword_percentage}")
            main_args.output = output_basename + f"_stop{int(stopword_percentage*100)}.res"
            main_args.stopwords = stopword_percentage
            main(main_args)
        exit(0)
    
    main(main_args)