"""
Cross-Validation Module
========================

Train/test cross-validation framework for multi-dataset analysis.

Classes:
    CrossValidator: Cross-dataset prediction and mutual information

Implements:
- ACT → Planck prediction accuracy
- Planck → ACT prediction accuracy
- Mutual information analysis
- Prediction quality metrics
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.stats import entropy

from .utils import OutputManager
from .phase_detector import PhaseTransitionDetector


class CrossValidator:
    """
    Cross-validation framework for CMB datasets.
    
    Tests predictive power: can detections in one dataset predict features
    in another independent dataset?
    
    Attributes:
        output (OutputManager): For logging
        resolution_ratio (float): Beam FWHM ratio for resolution correction
        
    Example:
        >>> validator = CrossValidator()
        >>> results = validator.run_all_tests(act_data, planck_data, act_transitions)
    """
    
    def __init__(self, output: OutputManager = None):
        """
        Initialize CrossValidator.
        
        Parameters:
            output (OutputManager, optional): For logging
        """
        self.output = output if output is not None else OutputManager()
        self.resolution_ratio = 5.21  # Planck/ACT default
    
    def predict_planck_from_act(self,
                                act_transitions: np.ndarray,
                                resolution_ratio: float) -> np.ndarray:
        """
        Predict Planck transition locations from ACT detections.
        
        Applies resolution correction: lower resolution may merge/smooth features.
        
        Physical basis:
        - Beam smoothing blurs features over Δℓ ≈ π/θ_beam
        - ACT (1.4'): Δℓ ≈ 250, Planck (7.3'): Δℓ ≈ 1300
        - Features separated by <1300 multipoles may merge in Planck
        
        Parameters:
            act_transitions (ndarray): Detected transitions in ACT
            resolution_ratio (float): Planck FWHM / ACT FWHM (≈5.21)
            
        Returns:
            ndarray: Predicted Planck transition locations
        """
        if len(act_transitions) == 0:
            return np.array([])
        
        # Effective resolution in multipole space
        # Planck smooths over ~1300 multipoles at these scales
        planck_resolution = 250 * resolution_ratio  # ≈1300 multipoles
        
        predicted = []
        i = 0
        while i < len(act_transitions):
            current = act_transitions[i]
            
            # Check if next features are within Planck resolution
            # If so, they may merge into single broader feature
            merged_features = [current]
            j = i + 1
            
            while j < len(act_transitions):
                separation = act_transitions[j] - current
                if separation < planck_resolution:
                    merged_features.append(act_transitions[j])
                    j += 1
                else:
                    break
            
            if len(merged_features) > 1:
                # Multiple ACT features within Planck resolution
                # Predict weighted centroid (intensity-weighted average)
                predicted.append(np.mean(merged_features))
                i = j
            else:
                # Single feature - expect at same location
                predicted.append(current)
                i += 1
        
        return np.array(predicted)
    
    def predict_act_from_planck(self,
                                planck_transitions: np.ndarray,
                                resolution_ratio: float) -> np.ndarray:
        """
        Predict ACT transition locations from Planck detections.
        
        Higher resolution may resolve blended features into multiple components.
        
        Physical basis:
        - Single broad feature in Planck may be multiple sharp features in ACT
        - However, without additional information, we cannot predict
          how many features or their exact locations
        - Conservative approach: expect features at Planck locations
          with resolution-scaled uncertainty
        
        Parameters:
            planck_transitions (ndarray): Detected transitions in Planck
            resolution_ratio (float): Planck FWHM / ACT FWHM (≈5.21)
            
        Returns:
            ndarray: Predicted ACT transition locations
        """
        if len(planck_transitions) == 0:
            return np.array([])
        
        # Conservative prediction: Planck features should appear in ACT
        # at similar locations, but ACT may resolve additional structure
        # We cannot predict sub-structure without the actual ACT data
        
        # Predict features at Planck locations
        # (ACT may detect more, but these should definitely appear)
        predicted_act = planck_transitions.copy()
        
        # Note: This prediction has intrinsic limitations
        # ACT may detect 2-3 features where Planck sees 1 blended feature
        # But without prior info, we can't predict the splitting pattern
        
        return predicted_act
    
    def match_predictions(self,
                         predictions: np.ndarray,
                         actual: np.ndarray,
                         tolerance: float) -> Tuple[int, int, float]:
        """
        Match predictions to actual detections.
        
        Parameters:
            predictions (ndarray): Predicted locations
            actual (ndarray): Actual detected locations
            tolerance (float): Matching tolerance in multipoles
            
        Returns:
            tuple: (n_matched, n_predictions, accuracy)
                - n_matched: Number of predictions confirmed
                - n_predictions: Total predictions
                - accuracy: Match rate
        """
        if len(predictions) == 0 or len(actual) == 0:
            return 0, len(predictions), 0.0
        
        n_matched = 0
        for pred in predictions:
            # Find closest actual detection
            distances = np.abs(actual - pred)
            min_dist = np.min(distances)
            
            if min_dist <= tolerance:
                n_matched += 1
        
        accuracy = n_matched / len(predictions) if len(predictions) > 0 else 0.0
        
        return n_matched, len(predictions), accuracy
    
    def compute_rmse(self,
                    predictions: np.ndarray,
                    actual: np.ndarray,
                    tolerance: float) -> float:
        """
        Compute RMSE for matched predictions.
        
        Parameters:
            predictions (ndarray): Predicted locations
            actual (ndarray): Actual locations
            tolerance (float): Matching tolerance
            
        Returns:
            float: RMSE of matched pairs
        """
        if len(predictions) == 0 or len(actual) == 0:
            return np.nan
        
        matched_pairs = []
        for pred in predictions:
            distances = np.abs(actual - pred)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            if min_dist <= tolerance:
                matched_pairs.append((pred, actual[min_idx]))
        
        if len(matched_pairs) == 0:
            return np.nan
        
        matched_pairs = np.array(matched_pairs)
        rmse = np.sqrt(np.mean((matched_pairs[:, 0] - matched_pairs[:, 1])**2))
        
        return rmse
    
    def run_act_to_planck_test(self,
                               act_transitions: np.ndarray,
                               planck_transitions: np.ndarray,
                               resolution_ratio: float = 5.21) -> Dict[str, Any]:
        """
        Test ACT → Planck prediction.
        
        Parameters:
            act_transitions (ndarray): Detected in ACT
            planck_transitions (ndarray): Detected in Planck
            resolution_ratio (float): Planck/ACT beam ratio
            
        Returns:
            dict: Prediction test results
        """
        self.output.log_message("\nACT → Planck Prediction Test:")
        self.output.log_message(f"  Training set (ACT): {len(act_transitions)} transitions")
        self.output.log_message(f"  Test set (Planck): {len(planck_transitions)} transitions")
        
        # Make predictions
        predictions = self.predict_planck_from_act(act_transitions, resolution_ratio)
        
        # Resolution-aware tolerance
        base_tolerance = 200  # Base tolerance in multipoles
        tolerance = base_tolerance * np.sqrt(resolution_ratio)  # Scale with resolution
        
        self.output.log_message(f"  Tolerance: {tolerance:.0f} multipoles")
        
        # Match predictions
        n_matched, n_pred, accuracy = self.match_predictions(
            predictions, planck_transitions, tolerance
        )
        
        # Compute RMSE
        rmse = self.compute_rmse(predictions, planck_transitions, tolerance)
        
        self.output.log_message(f"  Predictions: {n_pred}")
        self.output.log_message(f"  Matched: {n_matched}")
        self.output.log_message(f"  Accuracy: {accuracy*100:.1f}%")
        self.output.log_message(f"  RMSE: {rmse:.1f} multipoles" if not np.isnan(rmse) else "  RMSE: N/A")
        
        return {
            'direction': 'act_to_planck',
            'n_predictions': len(predictions),
            'n_actual': len(planck_transitions),
            'n_matched': int(n_matched),
            'accuracy': float(accuracy),
            'rmse': float(rmse) if not np.isnan(rmse) else None,
            'tolerance': float(tolerance)
        }
    
    def run_planck_to_act_test(self,
                               planck_transitions: np.ndarray,
                               act_transitions: np.ndarray,
                               resolution_ratio: float = 5.21) -> Dict[str, Any]:
        """
        Test Planck → ACT prediction.
        
        Parameters:
            planck_transitions (ndarray): Detected in Planck
            act_transitions (ndarray): Detected in ACT
            resolution_ratio (float): Planck/ACT beam ratio
            
        Returns:
            dict: Prediction test results
        """
        self.output.log_message("\nPlanck → ACT Prediction Test:")
        self.output.log_message(f"  Training set (Planck): {len(planck_transitions)} transitions")
        self.output.log_message(f"  Test set (ACT): {len(act_transitions)} transitions")
        
        # Make predictions
        predictions = self.predict_act_from_planck(planck_transitions, resolution_ratio)
        
        # Tolerance (ACT has higher resolution, use smaller tolerance)
        tolerance = 200  # Standard tolerance for high-res
        
        self.output.log_message(f"  Tolerance: {tolerance:.0f} multipoles")
        
        # Match predictions
        n_matched, n_pred, accuracy = self.match_predictions(
            predictions, act_transitions, tolerance
        )
        
        # Compute RMSE
        rmse = self.compute_rmse(predictions, act_transitions, tolerance)
        
        self.output.log_message(f"  Predictions: {n_pred}")
        self.output.log_message(f"  Matched: {n_matched}")
        self.output.log_message(f"  Accuracy: {accuracy*100:.1f}%")
        self.output.log_message(f"  RMSE: {rmse:.1f} multipoles" if not np.isnan(rmse) else "  RMSE: N/A")
        
        return {
            'direction': 'planck_to_act',
            'n_predictions': len(predictions),
            'n_actual': len(act_transitions),
            'n_matched': int(n_matched),
            'accuracy': float(accuracy),
            'rmse': float(rmse) if not np.isnan(rmse) else None,
            'tolerance': float(tolerance)
        }
    
    def compute_mutual_information(self,
                                  act_transitions: np.ndarray,
                                  planck_transitions: np.ndarray,
                                  ell_range: Tuple[float, float] = (500, 2000),
                                  n_bins: int = 10) -> float:
        """
        Compute mutual information between detection patterns.
        
        Discretizes multipole space and measures information shared
        between ACT and Planck detection patterns.
        
        Parameters:
            act_transitions (ndarray): ACT detections
            planck_transitions (ndarray): Planck detections
            ell_range (tuple): Multipole range to consider
            n_bins (int): Number of bins for discretization
            
        Returns:
            float: Mutual information in bits
        """
        # Create bins
        bins = np.linspace(ell_range[0], ell_range[1], n_bins+1)
        
        # Histogram each dataset
        hist_act, _ = np.histogram(act_transitions, bins=bins)
        hist_planck, _ = np.histogram(planck_transitions, bins=bins)
        
        # Convert to presence/absence (binary)
        present_act = (hist_act > 0).astype(int)
        present_planck = (hist_planck > 0).astype(int)
        
        # Joint distribution
        joint = np.zeros((2, 2))
        for i in range(n_bins):
            joint[present_act[i], present_planck[i]] += 1
        
        joint = joint / n_bins
        
        # Marginals
        p_act = joint.sum(axis=1)
        p_planck = joint.sum(axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(2):
            for j in range(2):
                if joint[i, j] > 0 and p_act[i] > 0 and p_planck[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (p_act[i] * p_planck[j]))
        
        return mi
    
    def run_all_tests(self,
                     act_data: Dict[str, np.ndarray],
                     planck_data: Dict[str, np.ndarray],
                     act_transitions: np.ndarray,
                     planck_transitions: np.ndarray) -> Dict[str, Any]:
        """
        Run complete cross-validation suite.
        
        Parameters:
            act_data (dict): ACT dataset
            planck_data (dict): Planck dataset
            act_transitions (ndarray): ACT detections
            planck_transitions (ndarray): Planck detections
            
        Returns:
            dict: Complete cross-validation results
        """
        self.output.log_subsection("CROSS-VALIDATION TESTS")
        
        # Test 1: ACT → Planck
        act_to_planck = self.run_act_to_planck_test(
            act_transitions, planck_transitions, self.resolution_ratio
        )
        
        # Test 2: Planck → ACT
        planck_to_act = self.run_planck_to_act_test(
            planck_transitions, act_transitions, self.resolution_ratio
        )
        
        # Mutual information
        mi = self.compute_mutual_information(act_transitions, planck_transitions)
        
        self.output.log_message(f"\nMutual Information: {mi:.3f} bits")
        if mi > 0.5:
            self.output.log_message("  ✓ High MI: Datasets seeing correlated features")
        elif mi > 0.2:
            self.output.log_message("  ~ Moderate MI: Some shared information")
        else:
            self.output.log_message("  ✗ Low MI: Datasets largely independent")
        
        # Combined prediction score
        combined_accuracy = np.sqrt(
            act_to_planck['accuracy'] * planck_to_act['accuracy']
        )
        
        self.output.log_message(f"\nCombined Prediction Score: {combined_accuracy*100:.1f}%")
        
        # Assessment
        self.output.log_message("\nCross-Validation Assessment:")
        if combined_accuracy >= 0.7 and mi >= 0.5:
            self.output.log_message("  ✓ STRONG: High prediction accuracy + high MI")
            assessment = "strong"
        elif combined_accuracy >= 0.5 and mi >= 0.2:
            self.output.log_message("  ~ MODERATE: Decent accuracy + moderate MI")
            assessment = "moderate"
        else:
            self.output.log_message("  ✗ WEAK: Low prediction power")
            assessment = "weak"
        
        return {
            'act_to_planck': act_to_planck,
            'planck_to_act': planck_to_act,
            'mutual_information': float(mi),
            'combined_accuracy': float(combined_accuracy),
            'assessment': assessment
        }

