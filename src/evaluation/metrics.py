"""
Evaluation Metrics for HS Code Classification
Comprehensive metrics for assessing model performance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HSClassificationMetrics:
    """Metrics for evaluating HS code classification"""
    
    @staticmethod
    def top_k_accuracy(
        predictions: List[List[str]],
        ground_truth: List[str],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Calculate Top-K accuracy
        
        Args:
            predictions: List of ranked predictions for each sample
                        [[pred1, pred2, pred3, ...], ...]
            ground_truth: List of true HS6 codes
            k_values: Values of K to evaluate
        
        Returns:
            Dictionary with Top-K accuracies
        """
        results = {}
        
        for k in k_values:
            correct = 0
            for preds, true_code in zip(predictions, ground_truth):
                if true_code in preds[:k]:
                    correct += 1
            
            accuracy = correct / len(ground_truth)
            results[f'top_{k}_accuracy'] = accuracy
        
        return results
    
    @staticmethod
    def mean_reciprocal_rank(
        predictions: List[List[str]],
        ground_truth: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        MRR = average of (1 / rank of first correct answer)
        
        Args:
            predictions: List of ranked predictions
            ground_truth: List of true HS6 codes
        
        Returns:
            MRR score (higher is better)
        """
        reciprocal_ranks = []
        
        for preds, true_code in zip(predictions, ground_truth):
            try:
                rank = preds.index(true_code) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                # True code not in predictions
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def hierarchical_accuracy(
        predictions_hs6: List[str],
        ground_truth_hs6: List[str]
    ) -> Dict[str, float]:
        """
        Calculate accuracy at each hierarchical level
        
        Args:
            predictions_hs6: Predicted HS6 codes
            ground_truth_hs6: True HS6 codes
        
        Returns:
            Dictionary with accuracy at Chapter, Heading, HS6 levels
        """
        # Extract hierarchical levels
        pred_chapters = [code[:2] for code in predictions_hs6]
        true_chapters = [code[:2] for code in ground_truth_hs6]
        
        pred_headings = [code[:4] for code in predictions_hs6]
        true_headings = [code[:4] for code in ground_truth_hs6]
        
        # Calculate accuracies
        chapter_acc = accuracy_score(true_chapters, pred_chapters)
        heading_acc = accuracy_score(true_headings, pred_headings)
        hs6_acc = accuracy_score(ground_truth_hs6, predictions_hs6)
        
        return {
            'chapter_accuracy': chapter_acc,
            'heading_accuracy': heading_acc,
            'hs6_accuracy': hs6_acc
        }
    
    @staticmethod
    def category_level_performance(
        predictions_hs6: List[str],
        ground_truth_hs6: List[str],
        descriptions: List[str]
    ) -> pd.DataFrame:
        """
        Analyze performance by HS chapter
        
        Args:
            predictions_hs6: Predicted HS6 codes
            ground_truth_hs6: True HS6 codes
            descriptions: Product descriptions
        
        Returns:
            DataFrame with per-chapter performance
        """
        # Extract chapters
        true_chapters = [code[:2] for code in ground_truth_hs6]
        
        # Create DataFrame
        df = pd.DataFrame({
            'true_chapter': true_chapters,
            'true_hs6': ground_truth_hs6,
            'pred_hs6': predictions_hs6,
            'description': descriptions
        })
        
        # Add correctness flag
        df['correct'] = df['true_hs6'] == df['pred_hs6']
        
        # Group by chapter
        chapter_stats = df.groupby('true_chapter').agg({
            'correct': ['count', 'sum', 'mean']
        })
        
        chapter_stats.columns = ['total_samples', 'correct_predictions', 'accuracy']
        chapter_stats = chapter_stats.reset_index()
        chapter_stats = chapter_stats.sort_values('accuracy', ascending=False)
        
        return chapter_stats
    
    @staticmethod
    def confusion_analysis(
        predictions_hs6: List[str],
        ground_truth_hs6: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Find most common confusion pairs
        
        Args:
            predictions_hs6: Predicted HS6 codes
            ground_truth_hs6: True HS6 codes
            top_n: Number of top confusions to return
        
        Returns:
            DataFrame with most common prediction errors
        """
        # Find misclassifications
        errors = [
            (true, pred) 
            for true, pred in zip(ground_truth_hs6, predictions_hs6)
            if true != pred
        ]
        
        # Count occurrences
        error_counts = pd.Series(errors).value_counts().head(top_n)
        
        # Format as DataFrame
        confusion_df = pd.DataFrame({
            'true_hs6': [pair[0] for pair in error_counts.index],
            'predicted_hs6': [pair[1] for pair in error_counts.index],
            'count': error_counts.values
        })
        
        return confusion_df
    
    @staticmethod
    def tariff_weighted_accuracy(
        predictions_hs6: List[str],
        ground_truth_hs6: List[str],
        tariff_rates: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate accuracy weighted by tariff impact
        
        Misclassifications causing larger tariff errors are penalized more
        
        Args:
            predictions_hs6: Predicted HS6 codes
            ground_truth_hs6: True HS6 codes
            tariff_rates: Dict mapping HS6 -> tariff rate (%)
        
        Returns:
            Dictionary with weighted metrics
        """
        total_tariff_error = 0
        total_samples = 0
        correct_predictions = 0
        
        for pred, true in zip(predictions_hs6, ground_truth_hs6):
            pred_rate = tariff_rates.get(pred, 0)
            true_rate = tariff_rates.get(true, 0)
            
            # Absolute difference in tariff rates
            tariff_error = abs(pred_rate - true_rate)
            total_tariff_error += tariff_error
            
            if pred == true:
                correct_predictions += 1
            
            total_samples += 1
        
        return {
            'standard_accuracy': correct_predictions / total_samples,
            'mean_tariff_error': total_tariff_error / total_samples,
            'zero_tariff_error_rate': correct_predictions / total_samples
        }


def evaluate_model(
    model,
    test_data: pd.DataFrame,
    top_k: int = 5
) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: Model with predict() method
        test_data: DataFrame with 'description' and 'hs6' columns
        top_k: Number of predictions to generate
    
    Returns:
        Dictionary with all metrics
    """
    logger.info("Running comprehensive evaluation...")
    
    # Generate predictions
    all_predictions = []
    all_ground_truth = test_data['hs6'].tolist()
    all_descriptions = test_data['description'].tolist()
    
    for desc in all_descriptions:
        results = model.predict(desc, top_k=top_k)
        pred_hs6 = results['hs6'].tolist()
        all_predictions.append(pred_hs6)
    
    # Calculate metrics
    metrics_calc = HSClassificationMetrics()
    
    results = {}
    
    # Top-K accuracy
    topk_results = metrics_calc.top_k_accuracy(
        all_predictions,
        all_ground_truth,
        k_values=[1, 3, 5]
    )
    results.update(topk_results)
    
    # MRR
    results['mrr'] = metrics_calc.mean_reciprocal_rank(
        all_predictions,
        all_ground_truth
    )
    
    # Hierarchical accuracy (using top-1 predictions)
    top1_predictions = [preds[0] for preds in all_predictions]
    hierarchical_results = metrics_calc.hierarchical_accuracy(
        top1_predictions,
        all_ground_truth
    )
    results.update(hierarchical_results)
    
    # Category-level performance
    category_perf = metrics_calc.category_level_performance(
        top1_predictions,
        all_ground_truth,
        all_descriptions
    )
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nTop 10 Performing Chapters:")
    print(category_perf.head(10))
    
    logger.info("\nBottom 10 Performing Chapters:")
    print(category_perf.tail(10))
    
    return {
        'metrics': results,
        'category_performance': category_perf
    }


if __name__ == "__main__":
    # Example usage
    logger.info("Evaluation module loaded.")
    logger.info("Use evaluate_model() function to assess model performance.")
