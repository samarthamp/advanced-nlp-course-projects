"""
Task 1: Evaluation Metrics Analysis
Implementation and demonstration of ROUGE, BLEU, BERT Score, and custom metric
"""

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import numpy as np

class EvaluationMetrics:
    """Comprehensive evaluation metrics toolkit for NLP tasks"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_stemmer=True
        )
    
    def calculate_rouge(self, reference, hypothesis):
        """
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
        
        Types:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest Common Subsequence
        
        Advantages:
        - Good for summarization tasks
        - Captures different levels of n-gram overlap
        - Widely used and understood
        
        Disadvantages:
        - Focuses on recall, may miss precision
        - Doesn't capture semantic meaning
        - Sensitive to word order changes
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        }
    
    def calculate_bleu(self, reference, hypothesis, max_n=4):
        """
        BLEU (Bilingual Evaluation Understudy)
        
        Types:
        - BLEU-1: Unigram precision
        - BLEU-2: Up to bigram precision
        - BLEU-3: Up to trigram precision
        - BLEU-4: Up to 4-gram precision
        
        Advantages:
        - Good for machine translation
        - Considers precision with brevity penalty
        - Language independent
        
        Disadvantages:
        - Focuses on precision (not recall)
        - Poor for single reference translations
        - Doesn't capture semantic similarity
        - Struggles with paraphrases
        """
        reference_tokens = [reference.split()]
        hypothesis_tokens = hypothesis.split()
        
        smoothing = SmoothingFunction()
        bleu_scores = {}
        
        for n in range(1, max_n + 1):
            weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
            bleu_scores[f'bleu_{n}'] = sentence_bleu(
                reference_tokens,
                hypothesis_tokens,
                weights=weights,
                smoothing_function=smoothing.method1
            )
        
        return bleu_scores
    
    def calculate_bert_score(self, references, hypotheses, lang='en'):
        """
        BERT Score - Contextualized embedding-based metric
        
        Advantages:
        - Captures semantic similarity
        - Robust to paraphrases
        - Contextual understanding
        - Correlates well with human judgment
        
        Disadvantages:
        - Computationally expensive
        - Requires pre-trained model
        - Can be slow for large datasets
        - Model-dependent results
        """
        P, R, F1 = bert_score(
            hypotheses,
            references,
            lang=lang,
            verbose=False
        )
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }


class SemanticCoherenceScore:
    """
    BONUS: Custom Evaluation Metric
    
    Semantic Coherence Score (SCS) combines:
    1. Semantic similarity (using sentence embeddings)
    2. Structural similarity (n-gram overlap)
    3. Fluency (length-based heuristic)
    
    Why it's better:
    - Captures both semantic AND structural similarity
    - Balances recall and precision
    - More robust to paraphrasing than ROUGE/BLEU
    - More efficient than BERT Score
    """
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_scs(self, reference, hypothesis, alpha=0.5, beta=0.3, gamma=0.2):
        """
        Calculate Semantic Coherence Score
        
        SCS = α * semantic_sim + β * structural_sim + γ * fluency_score
        """
        # 1. Semantic Similarity
        ref_embedding = self.embedding_model.encode(reference)
        hyp_embedding = self.embedding_model.encode(hypothesis)
        
        semantic_sim = np.dot(ref_embedding, hyp_embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(hyp_embedding)
        )
        
        # 2. Structural Similarity
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            structural_sim = 0
        else:
            intersection = len(ref_tokens.intersection(hyp_tokens))
            union = len(ref_tokens.union(hyp_tokens))
            structural_sim = intersection / union if union > 0 else 0
        
        # 3. Fluency Score
        avg_word_length = np.mean([len(w) for w in hypothesis.split()])
        fluency_score = min(1.0, avg_word_length / 6.0)
        
        # Calculate final SCS
        scs = alpha * semantic_sim + beta * structural_sim + gamma * fluency_score
        
        return {
            'scs': scs,
            'semantic_similarity': semantic_sim,
            'structural_similarity': structural_sim,
            'fluency_score': fluency_score
        }
    
    def explain_advantages(self):
        """Explain why SCS is better"""
        print("\n" + "="*70)
        print("SEMANTIC COHERENCE SCORE (SCS) - CUSTOM METRIC")
        print("="*70)
        print("\nAdvantages over ROUGE:")
        print("  ✓ Captures semantic meaning, not just word overlap")
        print("  ✓ More robust to paraphrasing")
        
        print("\nAdvantages over BLEU:")
        print("  ✓ Works well with single references")
        print("  ✓ Better handles synonyms")
        
        print("\nAdvantages over BERT Score:")
        print("  ✓ More computationally efficient")
        print("  ✓ Tunable weights for different tasks")


def demonstrate_metrics():
    """Demonstrate all evaluation metrics"""
    evaluator = EvaluationMetrics()
    
    reference = "The cat sat on the mat and looked very comfortable."
    hypothesis1 = "The cat was sitting on the mat looking comfortable."
    hypothesis2 = "A dog ran in the park."
    
    print("="*70)
    print("EVALUATION METRICS DEMONSTRATION")
    print("="*70)
    print("\nReference:", reference)
    print("Hypothesis 1 (similar):", hypothesis1)
    print("Hypothesis 2 (different):", hypothesis2)
    
    # ROUGE
    print("\n" + "="*70)
    print("ROUGE SCORES")
    print("="*70)
    rouge1 = evaluator.calculate_rouge(reference, hypothesis1)
    rouge2 = evaluator.calculate_rouge(reference, hypothesis2)
    
    print("\nHypothesis 1 (Similar):")
    for metric, scores in rouge1.items():
        print(f"  {metric.upper()}: P={scores['precision']:.3f}, R={scores['recall']:.3f}, F1={scores['f1']:.3f}")
    
    print("\nHypothesis 2 (Different):")
    for metric, scores in rouge2.items():
        print(f"  {metric.upper()}: P={scores['precision']:.3f}, R={scores['recall']:.3f}, F1={scores['f1']:.3f}")
    
    # BLEU
    print("\n" + "="*70)
    print("BLEU SCORES")
    print("="*70)
    bleu1 = evaluator.calculate_bleu(reference, hypothesis1)
    bleu2 = evaluator.calculate_bleu(reference, hypothesis2)
    
    print("\nHypothesis 1 (Similar):")
    for metric, score in bleu1.items():
        print(f"  {metric.upper()}: {score:.4f}")
    
    print("\nHypothesis 2 (Different):")
    for metric, score in bleu2.items():
        print(f"  {metric.upper()}: {score:.4f}")
    
    # BERT Score
    print("\n" + "="*70)
    print("BERT SCORE")
    print("="*70)
    bert1 = evaluator.calculate_bert_score([reference], [hypothesis1])
    bert2 = evaluator.calculate_bert_score([reference], [hypothesis2])
    
    print("\nHypothesis 1 (Similar):")
    print(f"  Precision: {bert1['precision']:.4f}")
    print(f"  Recall: {bert1['recall']:.4f}")
    print(f"  F1: {bert1['f1']:.4f}")
    
    print("\nHypothesis 2 (Different):")
    print(f"  Precision: {bert2['precision']:.4f}")
    print(f"  Recall: {bert2['recall']:.4f}")
    print(f"  F1: {bert2['f1']:.4f}")
    
    # Reference-Free
    print("\n" + "="*70)
    print("REFERENCE-FREE EVALUATION: PERPLEXITY")
    print("="*70)
    print("\nPerplexity measures text quality without needing a reference.")
    print("Lower perplexity = more fluent text (requires language model)")
    print("\nAdvantages:")
    print("  ✓ No reference needed")
    print("  ✓ Measures fluency and coherence")
    print("\nDisadvantages:")
    print("  ✗ Doesn't measure factual accuracy")
    print("  ✗ Can favor generic text")


if __name__ == '__main__':
    print("="*70)
    print("TASK 1: EVALUATION METRICS ANALYSIS")
    print("="*70)
    
    demonstrate_metrics()
    
    print("\n\n")
    custom_metric = SemanticCoherenceScore()
    custom_metric.explain_advantages()
    
    reference = "The cat sat on the mat and looked very comfortable."
    hypothesis = "A feline was resting comfortably on a rug."
    
    scores = custom_metric.calculate_scs(reference, hypothesis)
    
    print("\n" + "="*70)
    print("EXAMPLE: SCS Calculation")
    print("="*70)
    print(f"\nReference: {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"\nResults:")
    print(f"  Semantic Similarity: {scores['semantic_similarity']:.4f}")
    print(f"  Structural Similarity: {scores['structural_similarity']:.4f}")
    print(f"  Fluency Score: {scores['fluency_score']:.4f}")
    print(f"  → Final SCS: {scores['scs']:.4f}")
    
    print("\n✓ Task 1 completed successfully!")