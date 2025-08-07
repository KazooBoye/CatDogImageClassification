import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

class QuickNoiseDetector:
    """
    Quick and simple noise detection for cat/dog images
    Focuses on the most effective metrics for identifying problematic images
    """
    
    def __init__(self, dataset_path="Dataset"):
        self.dataset_path = dataset_path
        
    def analyze_single_image(self, image_path, show_analysis=True):
        """Analyze a single image and show detailed breakdown"""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate key metrics
        # 1. Blur detection (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Edge density (complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 3. Main object detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            total_area = gray.shape[0] * gray.shape[1]
            main_object_ratio = contour_area / total_area
        else:
            main_object_ratio = 0
        
        # 4. Background uniformity (check corners)
        h, w = gray.shape
        corner_size = min(h//6, w//6)
        corners = [
            gray[0:corner_size, 0:corner_size],                    # Top-left
            gray[0:corner_size, w-corner_size:w],                  # Top-right
            gray[h-corner_size:h, 0:corner_size],                  # Bottom-left
            gray[h-corner_size:h, w-corner_size:w]                 # Bottom-right
        ]
        corner_stds = [np.std(corner.astype(np.float32)) for corner in corners]
        background_noise = np.mean(corner_stds)
        
        # Calculate overall quality score (0-1, higher = better quality)
        # Normalize metrics
        blur_norm = min(blur_score / 1000, 1.0)  # Good if > 100
        edge_norm = min(edge_density * 10, 1.0)  # Bad if > 0.1  
        object_norm = min(main_object_ratio * 5, 1.0)  # Good if > 0.2
        background_norm = min(background_noise / 50, 1.0)  # Bad if > 30
        
        quality_score = (0.3 * blur_norm + 
                        0.2 * (1 - edge_norm) + 
                        0.3 * object_norm + 
                        0.2 * (1 - background_norm))
        
        metrics = {
            'blur_score': blur_score,
            'edge_density': edge_density,
            'main_object_ratio': main_object_ratio,
            'background_noise': background_noise,
            'quality_score': quality_score
        }
        
        if show_analysis:
            self.visualize_single_image_analysis(img, gray, edges, metrics, image_path)
        
        return metrics
    
    def visualize_single_image_analysis(self, img, gray, edges, metrics, image_path):
        """Create detailed visualization for a single image analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grayscale
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # Edges
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title(f'Edges (Density: {metrics["edge_density"]:.3f})')
        axes[0, 2].axis('off')
        
        # Corner analysis for background noise
        h, w = gray.shape
        corner_size = min(h//6, w//6)
        corners_img = gray.copy()
        cv2.rectangle(corners_img, (0, 0), (corner_size, corner_size), 128, 2)
        cv2.rectangle(corners_img, (w-corner_size, 0), (w, corner_size), 128, 2)
        cv2.rectangle(corners_img, (0, h-corner_size), (corner_size, h), 128, 2)
        cv2.rectangle(corners_img, (w-corner_size, h-corner_size), (w, h), 128, 2)
        
        axes[1, 0].imshow(corners_img, cmap='gray')
        axes[1, 0].set_title(f'Background Analysis\\n(Noise: {metrics["background_noise"]:.1f})')
        axes[1, 0].axis('off')
        
        # Metrics bar chart
        metric_names = ['Blur\\nScore', 'Edge\\nDensity', 'Object\\nRatio', 'Background\\nNoise']
        metric_values = [
            min(metrics['blur_score'] / 1000, 1.0),
            metrics['edge_density'] * 10,
            metrics['main_object_ratio'] * 5,
            metrics['background_noise'] / 50
        ]
        
        colors = ['green' if v < 0.5 else 'orange' if v < 0.8 else 'red' for v in metric_values]
        
        axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Quality Metrics\\n(Lower = Better except Blur)')
        axes[1, 1].set_ylim(0, 1.2)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Overall assessment
        quality = metrics['quality_score']
        if quality > 0.7:
            assessment = "HIGH QUALITY\\n✓ Good for training"
            color = 'green'
        elif quality > 0.4:
            assessment = "MEDIUM QUALITY\\n⚠ Review manually"
            color = 'orange'
        else:
            assessment = "LOW QUALITY\\n✗ Consider removing"
            color = 'red'
        
        axes[1, 2].text(0.5, 0.6, assessment, ha='center', va='center', 
                       fontsize=12, fontweight='bold', color=color,
                       transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.5, 0.3, f'Quality Score: {quality:.3f}', 
                       ha='center', va='center', fontsize=14,
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        filename = os.path.basename(image_path)
        plt.suptitle(f'Image Quality Analysis: {filename}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def quick_dataset_scan(self, max_images=50, save_results=True):
        """Quick scan of dataset to identify problematic images"""
        
        print("Quick Dataset Quality Scan")
        print("=" * 40)
        
        results = []
        
        for class_name in ['Cat', 'Dog']:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found")
                continue
                
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            files = files[:max_images//2]  # Split between cats and dogs
            
            print(f"\\nAnalyzing {len(files)} {class_name.lower()} images...")
            
            for i, filename in enumerate(files):
                filepath = os.path.join(class_path, filename)
                metrics = self.analyze_single_image(filepath, show_analysis=False)
                
                if metrics:
                    metrics['filename'] = filename
                    metrics['class'] = class_name.lower()
                    metrics['filepath'] = filepath
                    results.append(metrics)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(files)}")
        
        if not results:
            print("No images processed successfully")
            return None
        
        # Sort by quality score
        results_sorted = sorted(results, key=lambda x: x['quality_score'])
        
        # Display results
        print(f"\\n" + "=" * 60)
        print("QUALITY ANALYSIS RESULTS")
        print("=" * 60)
        
        print("\\n🔴 LOWEST QUALITY IMAGES (consider removing):")
        for i, img in enumerate(results_sorted[:10]):
            print(f"{i+1:2d}. {img['filename']:<20} ({img['class']}) - Score: {img['quality_score']:.3f}")
        
        print("\\n🟢 HIGHEST QUALITY IMAGES:")
        for i, img in enumerate(results_sorted[-10:]):
            print(f"{i+1:2d}. {img['filename']:<20} ({img['class']}) - Score: {img['quality_score']:.3f}")
        
        # Statistics
        scores = [r['quality_score'] for r in results]
        print(f"\\n📊 STATISTICS:")
        print(f"   Average quality score: {np.mean(scores):.3f}")
        print(f"   Median quality score: {np.median(scores):.3f}")
        print(f"   Std deviation: {np.std(scores):.3f}")
        
        low_quality_count = len([s for s in scores if s < 0.4])
        print(f"   Images below 0.4 quality: {low_quality_count}/{len(scores)} ({low_quality_count/len(scores)*100:.1f}%)")
        
        if save_results:
            # Create summary plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Quality score distribution
            axes[0].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
            axes[0].axvline(0.4, color='orange', linestyle='-', label='Low Quality Threshold')
            axes[0].set_xlabel('Quality Score')
            axes[0].set_ylabel('Number of Images')
            axes[0].set_title('Quality Score Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Class comparison
            cat_scores = [r['quality_score'] for r in results if r['class'] == 'cat']
            dog_scores = [r['quality_score'] for r in results if r['class'] == 'dog']
            
            axes[1].boxplot([cat_scores, dog_scores], labels=['Cats', 'Dogs'])
            axes[1].set_ylabel('Quality Score')
            axes[1].set_title('Quality Score by Class')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('dataset_quality_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\\n💾 Results saved to: dataset_quality_analysis.png")
        
        return results

def demo_single_image_analysis():
    """Demonstrate analysis on a few sample images"""
    detector = QuickNoiseDetector()
    
    # Find some sample images
    for class_name in ['Cat', 'Dog']:
        class_path = os.path.join(detector.dataset_path, class_name)
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                # Analyze first few images
                print(f"\\nAnalyzing sample {class_name.lower()} images:")
                for filename in files[:3]:  # Just first 3
                    filepath = os.path.join(class_path, filename)
                    print(f"\\nAnalyzing: {filename}")
                    detector.analyze_single_image(filepath, show_analysis=True)

def main():
    """Main function - run this to analyze your dataset"""
    detector = QuickNoiseDetector()
    
    print("Cat/Dog Dataset Quality Analyzer")
    print("=" * 40)
    
    choice = input("\\nChoose analysis type:\\n1. Quick dataset scan\\n2. Detailed single image analysis\\n3. Both\\nEnter choice (1-3): ")
    
    if choice in ['1', '3']:
        print("\\nStarting quick dataset scan...")
        detector.quick_dataset_scan(max_images=100)
    
    if choice in ['2', '3']:
        print("\\nRunning detailed single image analysis...")
        demo_single_image_analysis()
    
    print("\\nAnalysis complete!")

if __name__ == "__main__":
    main()
