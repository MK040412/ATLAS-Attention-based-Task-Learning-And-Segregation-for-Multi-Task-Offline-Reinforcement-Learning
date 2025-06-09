import torch
import math
from typing import List, Dict, Any

class DirichletProcessMM:
    """Variational Bayes Dirichlet Process Mixture Model (Eq. 5-7)"""
    def __init__(self, concentration: float, latent_dim: int, device: str = 'cpu'):
        self.alpha = concentration  # DP concentration parameter
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        # Cluster parameters (variational posteriors)
        self.cluster_means: List[torch.Tensor] = []  # μ_k
        self.cluster_covs: List[torch.Tensor] = []   # Σ_k
        self.cluster_counts: List[int] = []          # n_k
        self.cluster_weights: List[float] = []       # π_k

        # Hyperparameters for Gaussian priors
        self.prior_mean = torch.zeros(latent_dim, device=self.device)
        self.prior_cov = torch.eye(latent_dim, device=self.device)

        # Memoization for efficient inferences
        self.memo_assignments = {}
        self.memo_elbo = torch.tensor(0.0, device=self.device)
        print(f"🎯 DPMM initialized: α={concentration}, latent_dim={latent_dim}, device={self.device}")

    def infer_cluster(self, z: torch.Tensor) -> int:
        """Infer cluster assignment for single latent vector.
        Note: This is a simplified inference for single points, primarily for action selection.
        It does not update cluster statistics. For updates, use infer_batch or update_memo_vb.
        """
        if len(z.shape) == 1:
            z = z.unsqueeze(0) # Make it [1, latent_dim]
        
        z_on_dpmm_device = z.to(self.device) # Ensure z is on the same device as DPMM params

        if not self.cluster_means: # No clusters yet
            # Cannot assign to an existing cluster, so conceptually it's a new one.
            # However, we don't create it here to avoid side effects during pure inference.
            # The caller (e.g., agent) might decide to create one if needed or use a default.
            # For now, return a placeholder or handle appropriately.
            # Let's assume if no clusters, it would be assigned to cluster 0 if one were created.
            return 0 # Placeholder, or could raise an error / handle differently

        probs = self._compute_cluster_probabilities(z_on_dpmm_device.squeeze(0)) # Squeeze back to [latent_dim]
        
        # Argmax to pick the most likely cluster (including the "new cluster" probability)
        cluster_id = torch.argmax(probs).item()

        # If argmax selected the "new cluster" slot, it means it's most likely a new cluster.
        # We return this index. The actual creation happens in `_create_new_cluster`.
        # If it's an existing cluster, we return its ID.
        if cluster_id == len(self.cluster_means):
             # This indicates a new cluster is preferred.
             # For pure inference, we don't modify state.
             # The caller can decide if a new expert/cluster needs to be instantiated.
             # Returning len(self.cluster_means) signals this.
             # Or, assign to the closest existing one if no new cluster creation is allowed here.
             if len(self.cluster_means) > 0: # If there are existing clusters
                # Find closest existing cluster as a fallback if not creating new
                min_dist = float('inf')
                best_cluster_idx = 0
                for idx, mean_k in enumerate(self.cluster_means):
                    dist = torch.norm(z_on_dpmm_device.squeeze(0) - mean_k)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_idx = idx
                return best_cluster_idx
             else: # No clusters, and new cluster indicated
                return 0 # Default to 0
        return cluster_id


    def infer_batch(self, Z: torch.Tensor) -> torch.Tensor:
        """Infer cluster assignments for batch of latent vectors and update clusters."""
        batch_size = Z.shape[0]
        # Ensure Z is on the same device as DPMM parameters
        Z_on_dpmm_device = Z.to(self.device)
        assignments = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i, z_i in enumerate(Z_on_dpmm_device):
            if not self.cluster_means: # No clusters yet
                new_cluster_id = self._create_new_cluster(z_i)
                assignments[i] = new_cluster_id
            else:
                probs = self._compute_cluster_probabilities(z_i)
                # cluster_id = torch.multinomial(probs, 1).item() # Stochastic assignment
                cluster_id = torch.argmax(probs).item() # Deterministic assignment

                if cluster_id == len(self.cluster_means): # Index for new cluster
                    new_cluster_id = self._create_new_cluster(z_i)
                    assignments[i] = new_cluster_id
                else:
                    # Assign to existing cluster and update its statistics
                    self._update_cluster(cluster_id, z_i)
                    assignments[i] = cluster_id
        return assignments

    def _compute_cluster_probabilities(self, z: torch.Tensor) -> torch.Tensor:
        """Compute cluster assignment probabilities (CRP + likelihood) for a single point z."""
        num_clusters = len(self.cluster_means)
        # Probabilities will be stored on the same device as z
        log_probs = torch.full((num_clusters + 1,), -float('inf'), device=z.device)

        total_count = sum(self.cluster_counts) if self.cluster_counts else 0

        # Existing clusters
        for k in range(num_clusters):
            # CRP probability: P(z_i = k | z_<i, params) = n_k / (N - 1 + alpha)
            # For a single point, N-1 is total_count (if z is considered new)
            # If z is part of the batch being assigned, then N-1 is total_count -1
            # Let's use the standard CRP form: count_k / (total_points_so_far + alpha)
            # For numerical stability, ensure denominator is not zero.
            crp_prob_num = float(self.cluster_counts[k])
            crp_prob_den = float(total_count + self.alpha) # Using total_count before adding current point
            if crp_prob_den <= 1e-8: crp_prob_den = 1e-8 # Avoid division by zero

            crp_log_prob = torch.log(torch.tensor(crp_prob_num / crp_prob_den + 1e-8, device=z.device))

            mean_k = self.cluster_means[k].to(z.device) # Ensure mean is on z's device
            cov_k = self.cluster_covs[k].to(z.device)   # Ensure cov is on z's device
            # Add regularization to covariance for numerical stability
            cov_reg_k = cov_k + 1e-6 * torch.eye(self.latent_dim, device=z.device)

            try:
                likelihood = self._gaussian_logpdf(z, mean_k, cov_reg_k)
                log_probs[k] = crp_log_prob + likelihood
            except torch.linalg.LinAlgError: # Catch issues with Cholesky decomposition
                # Fallback if likelihood calculation fails (e.g. singular covariance)
                # Penalize this cluster heavily
                log_probs[k] = crp_log_prob - 50.0 # Large negative number
            except Exception:
                 log_probs[k] = crp_log_prob - 50.0


        # New cluster probability: P(z_i = new | z_<i, params) = alpha / (N - 1 + alpha)
        new_cluster_crp_prob_num = float(self.alpha)
        new_cluster_crp_prob_den = float(total_count + self.alpha)
        if new_cluster_crp_prob_den <= 1e-8: new_cluster_crp_prob_den = 1e-8

        new_cluster_crp_log_prob = torch.log(torch.tensor(new_cluster_crp_prob_num / new_cluster_crp_prob_den + 1e-8, device=z.device))
        
        prior_mean_dev = self.prior_mean.to(z.device)
        prior_cov_dev = self.prior_cov.to(z.device)
        try:
            prior_likelihood = self._gaussian_logpdf(z, prior_mean_dev, prior_cov_dev)
            log_probs[num_clusters] = new_cluster_crp_log_prob + prior_likelihood
        except torch.linalg.LinAlgError:
            log_probs[num_clusters] = new_cluster_crp_log_prob - 50.0
        except Exception:
            log_probs[num_clusters] = new_cluster_crp_log_prob - 50.0


        # Convert to probabilities using log_softmax for numerical stability
        probs = torch.softmax(log_probs, dim=0)
        
        # Final safety check to prevent NaN or Inf if all log_probs were -inf
        if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() < 1e-7:
            # Fallback to uniform if softmax fails
            probs = torch.ones(num_clusters + 1, device=z.device) / (num_clusters + 1)
        else:
            probs = torch.clamp(probs, 1e-8, 1.0) # Clamp probabilities
            probs = probs / probs.sum() # Renormalize

        return probs

    def _gaussian_logpdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Compute log probability density of multivariate Gaussian with numerical stability"""
        diff = x - mean
        
        # Ensure cov is positive definite for Cholesky decomposition
        # Add small jitter to diagonal
        cov_reg = cov + 1e-6 * torch.eye(self.latent_dim, device=x.device)
        
        try:
            L = torch.linalg.cholesky(cov_reg)
            # Solve L*y = diff for y, then L^T*alpha = y for alpha. alpha = cov_inv * diff
            # More stable: (diff^T * cov_inv * diff) = ||L^-1 * diff||^2
            solve_term = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
            quad_form = torch.sum(solve_term ** 2)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L))) # log(det(cov)) = log(det(L L^T)) = 2 * log(det(L))
            
            logpdf = -0.5 * (self.latent_dim * math.log(2 * math.pi) + logdet + quad_form)
        except torch.linalg.LinAlgError: # Fallback if Cholesky fails
            # Use a simpler diagonal covariance assumption or a fixed large variance
            # This is a fallback, so precision is less critical than avoiding a crash
            # Assuming diagonal covariance with small variance (e.g., 0.1 on diagonal)
            variance = 0.1
            logdet_fallback = self.latent_dim * math.log(variance)
            quad_form_fallback = torch.sum(diff ** 2) / variance
            logpdf = -0.5 * (self.latent_dim * math.log(2 * math.pi) + logdet_fallback + quad_form_fallback)
            # print(f"Warning: Cholesky decomposition failed in _gaussian_logpdf. Using fallback. Mean: {mean}, Cov diag: {torch.diag(cov)}")


        return torch.clamp(logpdf, -50.0, 50.0) # Clamp to avoid extreme values

    def _create_new_cluster(self, z: torch.Tensor) -> int:
        """Create new cluster with given point z (which is on self.device)."""
        cluster_id = len(self.cluster_means)
        self.cluster_means.append(z.clone().detach()) # Store on self.device
        # Initialize covariance as identity matrix or based on prior_cov
        self.cluster_covs.append(self.prior_cov.clone().detach()) # Store on self.device
        self.cluster_counts.append(1)
        # Weights will be recomputed
        self._recompute_weights()
        return cluster_id

    def _update_cluster(self, cluster_id: int, z: torch.Tensor):
        """Update cluster statistics with new point z (which is on self.device)."""
        if cluster_id >= len(self.cluster_means):
            # This should not happen if logic is correct
            print(f"Warning: Attempted to update non-existent cluster {cluster_id}")
            return

        # Update count
        old_count = self.cluster_counts[cluster_id]
        new_count = old_count + 1
        self.cluster_counts[cluster_id] = new_count

        # Update mean (online update Welford's algorithm style)
        old_mean = self.cluster_means[cluster_id]
        new_mean = old_mean + (z - old_mean) / float(new_count)
        self.cluster_means[cluster_id] = new_mean

        # Update covariance (simplified online update or recompute from scratch if needed)
        # For simplicity, a running average or a more robust online update like Welford's for covariance
        # A simple update:
        # diff_old = z - old_mean
        # diff_new = z - new_mean
        # self.cluster_covs[cluster_id] = ( (old_count -1)/new_count * self.cluster_covs[cluster_id] +
        #                                   (1/new_count) * torch.outer(diff_old, diff_new) )
        # Or, more simply, increase diagonal variance slightly or re-estimate if batch update
        # For now, let's use a simplified update that adds a bit of the new point's variance
        # This is a placeholder; a full Welford update for covariance is more robust.
        # Or, in M-step of VB, covariance is re-estimated from all points in cluster.
        # Here, we are doing sequential updates.
        # A common simplification is to make cov adapt slowly or re-estimate in M-step.
        # Let's assume M-step will handle full covariance update.
        # For now, just ensure it remains on the correct device.
        self.cluster_covs[cluster_id] = self.cluster_covs[cluster_id].to(self.device)


        self._recompute_weights()

    def _recompute_weights(self):
        """Recompute cluster weights based on counts."""
        if not self.cluster_counts:
            self.cluster_weights = []
            return
        total_count = sum(self.cluster_counts)
        if total_count == 0:
            self.cluster_weights = [1.0 / len(self.cluster_counts) if self.cluster_counts else 0.0] * len(self.cluster_counts)
        else:
            self.cluster_weights = [float(count) / total_count for count in self.cluster_counts]

    def update_memo_vb(self, Z: torch.Tensor, max_iterations: int = 10, tolerance: float = 1e-4) -> torch.Tensor:
        """Run Variational Bayes EM algorithm for DPMM parameter updates (Eq. 5-7)
        
        Args:
            Z: Batch of latent vectors [batch_size, latent_dim]
            max_iterations: Maximum number of VB iterations
            tolerance: Convergence tolerance for ELBO
            
        Returns:
            Final ELBO value
        """
        if Z.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
            
        # Ensure Z is on the correct device
        Z = Z.to(self.device)
        batch_size = Z.shape[0]
        
        # Initialize responsibilities if needed
        if len(self.cluster_means) == 0:
            # Create initial cluster with first point
            self._create_new_cluster(Z[0])
            
        num_clusters = len(self.cluster_means)
        prev_elbo = -float('inf')
        
        for iteration in range(max_iterations):
            # E-step: Compute responsibilities (variational posterior q(z_i))
            responsibilities = torch.zeros(batch_size, num_clusters + 1, device=self.device)
            
            for i, z_i in enumerate(Z):
                probs = self._compute_cluster_probabilities(z_i)
                responsibilities[i] = probs
                
            # M-step: Update cluster parameters using responsibilities
            self._update_cluster_parameters_vb(Z, responsibilities)
            
            # Compute ELBO for convergence check
            current_elbo = self._compute_elbo_vb(Z, responsibilities)
            
            # Check convergence
            if abs(current_elbo - prev_elbo) < tolerance:
                print(f"VB converged after {iteration + 1} iterations, ELBO: {current_elbo:.4f}")
                break
                
            prev_elbo = current_elbo
            
        # Handle birth/death moves for clusters
        self._birth_death_moves(Z, responsibilities)
        
        # Update memoized values
        self.memo_elbo = current_elbo
        
        return current_elbo

    def _update_cluster_parameters_vb(self, Z: torch.Tensor, responsibilities: torch.Tensor):
        """Update cluster parameters using VB M-step"""
        batch_size, num_clusters_plus_one = responsibilities.shape
        num_existing_clusters = len(self.cluster_means)
        
        # Update existing clusters
        for k in range(num_existing_clusters):
            r_k = responsibilities[:, k]  # Responsibilities for cluster k
            N_k = torch.sum(r_k)  # Effective number of points in cluster k
            
            if N_k > 1e-8:  # Only update if cluster has sufficient responsibility
                # Update cluster mean (weighted average)
                weighted_sum = torch.sum(r_k.unsqueeze(1) * Z, dim=0)
                new_mean = weighted_sum / N_k
                self.cluster_means[k] = new_mean
                
                # Update cluster covariance (weighted sample covariance)
                centered_Z = Z - new_mean.unsqueeze(0)
                weighted_outer = torch.sum(r_k.unsqueeze(1).unsqueeze(2) * 
                                         torch.bmm(centered_Z.unsqueeze(2), centered_Z.unsqueeze(1)), dim=0)
                new_cov = weighted_outer / N_k
                
                # Add regularization for numerical stability
                new_cov += 1e-6 * torch.eye(self.latent_dim, device=self.device)
                self.cluster_covs[k] = new_cov
                
                # Update count (for CRP probabilities)
                self.cluster_counts[k] = int(N_k.item())
            else:
                # Mark cluster for potential removal
                self.cluster_counts[k] = 0
                
        # Handle new cluster formation
        new_cluster_responsibility = responsibilities[:, -1]  # Last column is for new cluster
        N_new = torch.sum(new_cluster_responsibility)
        
        if N_new > 0.5:  # Threshold for creating new cluster
            # Create new cluster with points weighted by new cluster responsibility
            weighted_sum = torch.sum(new_cluster_responsibility.unsqueeze(1) * Z, dim=0)
            new_mean = weighted_sum / N_new
            
            # Initialize new cluster
            new_cluster_id = self._create_new_cluster(new_mean)
            
            # Update covariance for new cluster
            centered_Z = Z - new_mean.unsqueeze(0)
            weighted_outer = torch.sum(new_cluster_responsibility.unsqueeze(1).unsqueeze(2) * 
                                     torch.bmm(centered_Z.unsqueeze(2), centered_Z.unsqueeze(1)), dim=0)
            new_cov = weighted_outer / N_new
            new_cov += 1e-6 * torch.eye(self.latent_dim, device=self.device)
            self.cluster_covs[new_cluster_id] = new_cov
            self.cluster_counts[new_cluster_id] = int(N_new.item())
            
        # Remove empty clusters
        self._remove_empty_clusters()
        
        # Recompute weights
        self._recompute_weights()

    def _compute_elbo_vb(self, Z: torch.Tensor, responsibilities: torch.Tensor) -> torch.Tensor:
        """Compute Evidence Lower BOund for VB (Eq. 5-7)"""
        batch_size = Z.shape[0]
        num_clusters = len(self.cluster_means)
        
        if num_clusters == 0:
            return torch.tensor(0.0, device=self.device)
            
        elbo = torch.tensor(0.0, device=self.device)
        
        # Likelihood term: E[log p(X|Z,θ)]
        for i, z_i in enumerate(Z):
            for k in range(num_clusters):
                r_ik = responsibilities[i, k]
                if r_ik > 1e-8:
                    mean_k = self.cluster_means[k]
                    cov_k = self.cluster_covs[k]
                    log_likelihood = self._gaussian_logpdf(z_i, mean_k, cov_k)
                    elbo += r_ik * log_likelihood
                    
        # Prior term: E[log p(Z)] - Chinese Restaurant Process prior
        for k in range(num_clusters):
            N_k = torch.sum(responsibilities[:, k])
            if N_k > 1e-8:
                # CRP probability for cluster k
                if k == 0:
                    crp_log_prob = torch.log(torch.tensor(self.alpha, device=self.device))
                else:
                    total_prev = torch.sum(torch.tensor([self.cluster_counts[j] for j in range(k)], 
                                                      device=self.device, dtype=torch.float))
                    crp_log_prob = torch.log(torch.tensor(self.cluster_counts[k], device=self.device, dtype=torch.float) / 
                                           (total_prev + self.alpha))
                elbo += N_k * crp_log_prob
                
        # Entropy term: -E[log q(Z)]
        for i in range(batch_size):
            for k in range(num_clusters + 1):  # +1 for new cluster
                r_ik = responsibilities[i, k]
                if r_ik > 1e-8:
                    elbo -= r_ik * torch.log(r_ik + 1e-8)
                    
        return elbo

    def _birth_death_moves(self, Z: torch.Tensor, responsibilities: torch.Tensor):
        """Handle cluster birth and death moves based on responsibilities"""
        # Death moves: remove clusters with very low total responsibility
        clusters_to_remove = []
        for k in range(len(self.cluster_means)):
            total_resp = torch.sum(responsibilities[:, k])
            if total_resp < 0.1:  # Threshold for cluster death
                clusters_to_remove.append(k)
                
        # Remove clusters (in reverse order to maintain indices)
        for k in reversed(clusters_to_remove):
            self._remove_cluster(k)
            print(f"🪦 Removed cluster {k} (low responsibility)")
            
        # Birth moves are handled in _update_cluster_parameters_vb
        
    def _remove_empty_clusters(self):
        """Remove clusters with zero or very low counts"""
        clusters_to_remove = []
        for k in range(len(self.cluster_counts)):
            if self.cluster_counts[k] < 1:
                clusters_to_remove.append(k)
                
        # Remove in reverse order
        for k in reversed(clusters_to_remove):
            self._remove_cluster(k)
            
    def _remove_cluster(self, cluster_id: int):
        """Remove cluster at given index"""
        if cluster_id < len(self.cluster_means):
            del self.cluster_means[cluster_id]
            del self.cluster_covs[cluster_id]
            del self.cluster_counts[cluster_id]
            if cluster_id < len(self.cluster_weights):
                del self.cluster_weights[cluster_id]
                
        # Recompute weights after removal
        self._recompute_weights()

    def prune_clusters(self, min_points: int = 5) -> int:
        """Prune clusters with insufficient data points
        
        Args:
            min_points: Minimum number of points required for a cluster to survive
            
        Returns:
            Number of clusters removed
        """
        initial_count = len(self.cluster_means)
        clusters_to_remove = []
        
        for k in range(len(self.cluster_counts)):
            if self.cluster_counts[k] < min_points:
                clusters_to_remove.append(k)
                
        # Remove clusters in reverse order to maintain indices
        for k in reversed(clusters_to_remove):
            self._remove_cluster(k)
            
        removed_count = len(clusters_to_remove)
        if removed_count > 0:
            print(f"🧹 Pruned {removed_count} clusters with < {min_points} points")
            
        return removed_count

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        return {
            'num_clusters': len(self.cluster_means),
            'cluster_counts': self.cluster_counts.copy(),
            'cluster_weights': self.cluster_weights.copy(),
            'total_points': sum(self.cluster_counts) if self.cluster_counts else 0,
            'alpha': self.alpha,
            'latest_elbo': self.memo_elbo.item() if torch.is_tensor(self.memo_elbo) else self.memo_elbo
        }

    def reset_clusters(self):
        """Reset all clusters (useful for reinitialization)"""
        self.cluster_means.clear()
        self.cluster_covs.clear()
        self.cluster_counts.clear()
        self.cluster_weights.clear()
        self.memo_assignments.clear()
        self.memo_elbo = torch.tensor(0.0, device=self.device)
        print("🔄 All clusters reset")
