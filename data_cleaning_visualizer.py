import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.cluster import KMeans
from skimage import measure, segmentation, filters, morphology
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataCleaningVisualizer:
    """
    A comprehensive tool to visualize and analyze image noise in cat/dog datasets
    """
    
    def __init__(self, dataset_path="Dataset"):
        self.dataset_path = dataset_path
        self.cat_path = os.path.join(dataset_path, "Cat")
        self.dog_path = os.path.join(dataset_path, "Dog")
        self.analysis_results = {}
        
    def calculate_noise_metrics(self, image_path):
        """Calculate various noise and quality metrics for an image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Edge density (high edge density = complex/noisy image)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 2. Variance of Laplacian (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 3. Connected components analysis
            num_labels, labels = cv2.connectedComponents(edges)
            
            # Find largest component ratio
            largest_component = 0
            if num_labels > 1:
                for i in range(1, num_labels):
                    component_size = np.sum(labels == i)
                    largest_component = max(largest_component, component_size)
            
            main_object_ratio = largest_component / (gray.shape[0] * gray.shape[1])
            
            # 4. Color complexity (number of dominant colors)
            img_resized = cv2.resize(img, (50, 50))
            pixels = img_resized.reshape(-1, 3)
            
            # Use KMeans to find dominant colors
            try:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                kmeans.fit(pixels)
                color_complexity = len(np.unique(kmeans.labels_))
            except:
                color_complexity = 5
            
            # 5. Background uniformity
            # Check corners for background consistency
            h, w = gray.shape
            corners = [
                gray[0:h//4, 0:w//4],           # Top-left
                gray[0:h//4, 3*w//4:w],         # Top-right  
                gray[3*h//4:h, 0:w//4],         # Bottom-left
                gray[3*h//4:h, 3*w//4:w]        # Bottom-right
            ]
            
            corner_stds = [np.std(corner) for corner in corners]
            background_uniformity = np.mean(corner_stds)
            
            # 6. Texture analysis using Local Binary Pattern-like approach
            def local_texture_variance(img):
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                texture = cv2.filter2D(img, -1, kernel)
                return np.var(texture)
            
            texture_variance = local_texture_variance(gray)
            
            return {
                'edge_density': edge_density,
                'blur_score': laplacian_var,
                'main_object_ratio': main_object_ratio,
                'color_complexity': color_complexity,
                'background_uniformity': background_uniformity,
                'texture_variance': texture_variance,
                'num_components': num_labels
            }
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def analyze_dataset(self, max_images_per_class=500):
        """Analyze the entire dataset and collect noise metrics"""
        print("Analyzing dataset for noise metrics...")
        
        cat_metrics = []
        dog_metrics = []
        
        # Analyze cat images
        if os.path.exists(self.cat_path):
            cat_files = [f for f in os.listdir(self.cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            cat_files = cat_files[:max_images_per_class]  # Limit for performance
            
            print(f"Analyzing {len(cat_files)} cat images...")
            for i, filename in enumerate(cat_files):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(cat_files)}")
                    
                filepath = os.path.join(self.cat_path, filename)
                metrics = self.calculate_noise_metrics(filepath)
                if metrics:
                    metrics['filename'] = filename
                    metrics['class'] = 'cat'
                    metrics['filepath'] = filepath
                    cat_metrics.append(metrics)
        
        # Analyze dog images
        if os.path.exists(self.dog_path):
            dog_files = [f for f in os.listdir(self.dog_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            dog_files = dog_files[:max_images_per_class]  # Limit for performance
            
            print(f"Analyzing {len(dog_files)} dog images...")
            for i, filename in enumerate(dog_files):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(dog_files)}")
                    
                filepath = os.path.join(self.dog_path, filename)
                metrics = self.calculate_noise_metrics(filepath)
                if metrics:
                    metrics['filename'] = filename
                    metrics['class'] = 'dog'
                    metrics['filepath'] = filepath
                    dog_metrics.append(metrics)
        
        # Combine results
        all_metrics = cat_metrics + dog_metrics
        self.analysis_results = pd.DataFrame(all_metrics)
        
        print(f"Analysis complete! Processed {len(all_metrics)} images.")
        return self.analysis_results
    
    def calculate_noise_scores(self):
        """Calculate composite noise scores for each image"""
        if self.analysis_results.empty:
            print("No analysis results found. Run analyze_dataset() first.")
            return
        
        df = self.analysis_results.copy()
        
        # Normalize metrics to 0-1 scale
        metrics_to_normalize = ['edge_density', 'blur_score', 'main_object_ratio', 
                               'color_complexity', 'background_uniformity', 'texture_variance']
        
        for metric in metrics_to_normalize:
            df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        
        # Calculate composite noise score (higher = more noisy)
        # Bad indicators (higher = worse): edge_density, color_complexity, background_uniformity, texture_variance
        # Good indicators (higher = better): blur_score, main_object_ratio
        
        df['noise_score'] = (
            0.3 * df['edge_density_norm'] +           # High edge density = complex background
            0.2 * (1 - df['blur_score_norm']) +       # Low blur score = blurry image
            0.3 * (1 - df['main_object_ratio_norm']) + # Low main object ratio = small subject
            0.1 * df['color_complexity_norm'] +       # High color complexity = busy background
            0.1 * df['background_uniformity_norm']     # High background variance = noisy background
        )
        
        self.analysis_results = df
        return df
    
    def visualize_metrics_distribution(self):
        """Create comprehensive visualization of noise metrics"""
        if self.analysis_results.empty:
            print("No analysis results found. Run analyze_dataset() first.")
            return
        
        df = self.analysis_results
        
        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Image Quality Metrics Distribution Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['edge_density', 'blur_score', 'main_object_ratio', 
                  'color_complexity', 'background_uniformity', 'texture_variance', 'noise_score']
        
        # Plot distributions for each metric
        for i, metric in enumerate(metrics):
            if i < 9:
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # Box plot comparing cats vs dogs
                if metric in df.columns:
                    sns.boxplot(data=df, x='class', y=metric, ax=ax)
                    ax.set_title(f'{metric.replace("_", " ").title()}')
                    ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < 9:
            axes[2, 2].remove()
        
        # Add histogram for noise score
        axes[2, 1].hist(df[df['class'] == 'cat']['noise_score'], alpha=0.7, label='Cats', bins=30)
        axes[2, 1].hist(df[df['class'] == 'dog']['noise_score'], alpha=0.7, label='Dogs', bins=30)
        axes[2, 1].set_title('Noise Score Distribution')
        axes[2, 1].set_xlabel('Noise Score (higher = noisier)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('noise_metrics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def identify_outliers(self, threshold_percentile=90):
        """Identify the noisiest images for manual inspection"""
        if 'noise_score' not in self.analysis_results.columns:
            self.calculate_noise_scores()
        
        df = self.analysis_results
        threshold = np.percentile(df['noise_score'], threshold_percentile)
        
        noisy_images = df[df['noise_score'] >= threshold].sort_values('noise_score', ascending=False)
        
        print(f"\nTop {len(noisy_images)} noisiest images (>{threshold_percentile}th percentile):")
        print("=" * 80)
        
        for idx, row in noisy_images.head(20).iterrows():  # Show top 20
            print(f"File: {row['filename']:<25} Class: {row['class']:<5} "
                  f"Noise Score: {row['noise_score']:.3f}")
            print(f"  Edge Density: {row['edge_density']:.3f}, "
                  f"Main Object Ratio: {row['main_object_ratio']:.3f}, "
                  f"Blur Score: {row['blur_score']:.1f}")
            print()
        
        return noisy_images
    
    def visualize_sample_images(self, num_clean=6, num_noisy=6):
        """Visualize sample clean and noisy images side by side"""
        if 'noise_score' not in self.analysis_results.columns:
            self.calculate_noise_scores()
        
        df = self.analysis_results
        
        # Get cleanest and noisiest images
        clean_images = df.nsmallest(num_clean, 'noise_score')
        noisy_images = df.nlargest(num_noisy, 'noise_score')
        
        fig, axes = plt.subplots(2, max(num_clean, num_noisy), figsize=(20, 8))
        
        # Plot clean images
        for i in range(num_clean):
            if i < len(clean_images):
                row = clean_images.iloc[i]
                img = cv2.imread(row['filepath'])
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[0, i].imshow(img_rgb)
                    axes[0, i].set_title(f"CLEAN: {row['filename'][:15]}...\n"
                                       f"Score: {row['noise_score']:.3f}", fontsize=10)
                    axes[0, i].axis('off')
        
        # Plot noisy images
        for i in range(num_noisy):
            if i < len(noisy_images):
                row = noisy_images.iloc[i]
                img = cv2.imread(row['filepath'])
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[1, i].imshow(img_rgb)
                    axes[1, i].set_title(f"NOISY: {row['filename'][:15]}...\n"
                                       f"Score: {row['noise_score']:.3f}", fontsize=10, color='red')
                    axes[1, i].axis('off')
        
        # Remove empty subplots
        for i in range(max(num_clean, num_noisy)):
            if i >= num_clean:
                axes[0, i].axis('off')
            if i >= num_noisy:
                axes[1, i].axis('off')
        
        plt.suptitle('Clean vs Noisy Images Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('clean_vs_noisy_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_cleaning_recommendations(self, noise_threshold=0.7):
        """Generate recommendations for data cleaning"""
        if 'noise_score' not in self.analysis_results.columns:
            self.calculate_noise_scores()
        
        df = self.analysis_results
        
        # Images to remove
        images_to_remove = df[df['noise_score'] >= noise_threshold]
        
        print(f"\n{'='*60}")
        print("DATA CLEANING RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Total images analyzed: {len(df)}")
        print(f"Images recommended for removal: {len(images_to_remove)} ({len(images_to_remove)/len(df)*100:.1f}%)")
        print(f"Images to keep: {len(df) - len(images_to_remove)} ({(len(df) - len(images_to_remove))/len(df)*100:.1f}%)")
        
        # Breakdown by class
        cat_remove = len(images_to_remove[images_to_remove['class'] == 'cat'])
        dog_remove = len(images_to_remove[images_to_remove['class'] == 'dog'])
        cat_total = len(df[df['class'] == 'cat'])
        dog_total = len(df[df['class'] == 'dog'])
        
        print(f"\nBreakdown by class:")
        print(f"  Cats: Remove {cat_remove}/{cat_total} ({cat_remove/cat_total*100:.1f}%)")
        print(f"  Dogs: Remove {dog_remove}/{dog_total} ({dog_remove/dog_total*100:.1f}%)")
        
        # Top reasons for removal
        print(f"\nMain quality issues identified:")
        high_edge = len(df[df['edge_density'] > df['edge_density'].quantile(0.8)])
        low_blur = len(df[df['blur_score'] < df['blur_score'].quantile(0.2)])
        low_main_object = len(df[df['main_object_ratio'] < df['main_object_ratio'].quantile(0.2)])
        
        print(f"  High background complexity: {high_edge} images")
        print(f"  Low image sharpness: {low_blur} images")
        print(f"  Small main subject: {low_main_object} images")
        
        return images_to_remove
    
    def create_cleaning_script(self, noise_threshold=0.7, output_file="clean_dataset.py"):
        """Generate a Python script to automatically clean the dataset"""
        images_to_remove = self.generate_cleaning_recommendations(noise_threshold)
        
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated data cleaning script
Generated on: {pd.Timestamp.now()}
Noise threshold used: {noise_threshold}
Images to remove: {len(images_to_remove)}
"""

import os
import shutil
from pathlib import Path

def clean_dataset():
    """Remove noisy images identified by the data cleaning analysis"""
    
    # Create backup directory
    backup_dir = "Dataset_backup"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        os.makedirs(os.path.join(backup_dir, "Cat"))
        os.makedirs(os.path.join(backup_dir, "Dog"))
        print(f"Created backup directory: {{backup_dir}}")
    
    # List of files to remove
    files_to_remove = [
'''
        
        # Add file paths to remove
        for idx, row in images_to_remove.iterrows():
            script_content += f'        "{row["filepath"]}",\n'
        
        script_content += '''    ]
    
    removed_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            # Create backup
            relative_path = os.path.relpath(file_path, "Dataset")
            backup_path = os.path.join(backup_dir, relative_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(file_path, backup_path)
            
            # Remove original
            os.remove(file_path)
            removed_count += 1
            print(f"Removed: {file_path}")
    
    print(f"\\nCleaning complete!")
    print(f"Removed {removed_count} noisy images")
    print(f"Backup created in: {backup_dir}")

if __name__ == "__main__":
    clean_dataset()
'''
        
        # Save script
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        print(f"\\nCleaning script saved as: {output_file}")
        print(f"To clean your dataset, run: python {output_file}")
        
        return output_file

def main():
    """Main function to run the complete data cleaning analysis"""
    print("Starting Data Cleaning Visualization and Analysis")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = DataCleaningVisualizer("Dataset")
    
    # Run analysis
    print("\\n1. Analyzing dataset...")
    results = visualizer.analyze_dataset(max_images_per_class=1000)  # Adjust as needed
    
    if results.empty:
        print("No images found or analysis failed.")
        return
    
    # Calculate noise scores
    print("\\n2. Calculating noise scores...")
    visualizer.calculate_noise_scores()
    
    # Create visualizations
    print("\\n3. Creating visualizations...")
    visualizer.visualize_metrics_distribution()
    visualizer.visualize_sample_images()
    
    # Identify outliers
    print("\\n4. Identifying outliers...")
    noisy_images = visualizer.identify_outliers(threshold_percentile=85)
    
    # Generate recommendations
    print("\\n5. Generating cleaning recommendations...")
    visualizer.generate_cleaning_recommendations(noise_threshold=0.65)
    
    # Create cleaning script
    print("\\n6. Creating automated cleaning script...")
    visualizer.create_cleaning_script(noise_threshold=0.65)
    
    print("\\n" + "=" * 60)
    print("Analysis complete! Check the generated visualizations and cleaning script.")
    print("Files created:")
    print("  - noise_metrics_analysis.png")
    print("  - clean_vs_noisy_samples.png") 
    print("  - clean_dataset.py")

if __name__ == "__main__":
    main()
