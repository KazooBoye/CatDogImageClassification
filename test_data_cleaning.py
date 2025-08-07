#!/usr/bin/env python3
"""
Simple demonstration of the data cleaning strategy
Run this to see how the noise detection works on your dataset
"""

import os
import sys
from quick_noise_detector import QuickNoiseDetector

def test_data_cleaning():
    """Test the data cleaning visualization on your dataset"""
    
    print("Testing Data Cleaning Strategy")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "Dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        print("Please ensure your dataset is in the 'Dataset' folder with 'Cat' and 'Dog' subfolders.")
        return False
    
    # Check subfolders
    cat_path = os.path.join(dataset_path, "Cat")
    dog_path = os.path.join(dataset_path, "Dog")
    
    if not os.path.exists(cat_path):
        print(f"❌ Cat folder '{cat_path}' not found!")
        return False
        
    if not os.path.exists(dog_path):
        print(f"❌ Dog folder '{dog_path}' not found!")
        return False
    
    # Count images
    cat_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    dog_files = [f for f in os.listdir(dog_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"✅ Dataset found!")
    print(f"   Cat images: {len(cat_files)}")
    print(f"   Dog images: {len(dog_files)}")
    print(f"   Total images: {len(cat_files) + len(dog_files)}")
    
    # Initialize detector
    detector = QuickNoiseDetector(dataset_path)
    
    # Run quick analysis
    print(f"\n🔍 Running quality analysis on sample images...")
    try:
        results = detector.quick_dataset_scan(max_images=20, save_results=True)
        
        if results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"   Analyzed {len(results)} images")
            print(f"   Results saved to 'dataset_quality_analysis.png'")
            
            # Show specific recommendations
            low_quality = [r for r in results if r['quality_score'] < 0.4]
            if low_quality:
                print(f"\n⚠️  Found {len(low_quality)} low-quality images that should be reviewed:")
                for img in low_quality[:5]:  # Show first 5
                    print(f"   - {img['filename']} (score: {img['quality_score']:.3f})")
                if len(low_quality) > 5:
                    print(f"   ... and {len(low_quality) - 5} more")
            else:
                print(f"\n✅ No obviously low-quality images found in sample!")
                
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def test_single_image():
    """Test detailed analysis on a single image"""
    
    dataset_path = "Dataset"
    
    # Find a sample image
    for class_name in ['Cat', 'Dog']:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                sample_file = os.path.join(class_path, files[0])
                
                print(f"\n🔬 Detailed analysis of sample image: {files[0]}")
                print("-" * 50)
                
                detector = QuickNoiseDetector(dataset_path)
                try:
                    metrics = detector.analyze_single_image(sample_file, show_analysis=True)
                    
                    if metrics:
                        print(f"✅ Analysis completed!")
                        print(f"   Quality Score: {metrics['quality_score']:.3f}")
                        print(f"   Blur Score: {metrics['blur_score']:.1f}")
                        print(f"   Edge Density: {metrics['edge_density']:.3f}")
                        print(f"   Main Object Ratio: {metrics['main_object_ratio']:.3f}")
                        print(f"   Background Noise: {metrics['background_noise']:.1f}")
                        
                        if metrics['quality_score'] > 0.7:
                            print("   🟢 HIGH QUALITY - Good for training")
                        elif metrics['quality_score'] > 0.4:
                            print("   🟡 MEDIUM QUALITY - Review manually") 
                        else:
                            print("   🔴 LOW QUALITY - Consider removing")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Error analyzing image: {e}")
                    return False
    
    print("❌ No sample images found!")
    return False

def main():
    """Main demonstration function"""
    
    print("🔍 Data Cleaning Strategy Demonstration")
    print("=" * 60)
    print()
    print("This script will:")
    print("1. Check your dataset structure")
    print("2. Analyze sample images for quality issues")
    print("3. Generate recommendations for data cleaning")
    print("4. Create visualizations to help you identify noisy images")
    print()
    
    # Test basic functionality
    success = test_data_cleaning()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("=" * 60)
        print()
        print("1. Review the generated 'dataset_quality_analysis.png' chart")
        print("2. Run the full analysis script for your entire dataset:")
        print("   python data_cleaning_visualizer.py")
        print()
        print("3. For detailed single-image analysis:")
        print("   python quick_noise_detector.py")
        print()
        print("4. The analysis will help you identify:")
        print("   - Blurry images")
        print("   - Images with complex/noisy backgrounds") 
        print("   - Images where the cat/dog is too small")
        print("   - Images with poor contrast")
        print()
        print("5. Use the generated cleaning recommendations to improve")
        print("   your model's training data quality!")
        
        # Optional: Show detailed analysis of one image
        print("\n" + "-" * 40)
        show_detail = input("Would you like to see detailed analysis of a sample image? (y/n): ")
        if show_detail.lower() == 'y':
            test_single_image()
    
    else:
        print("\n❌ Setup issue detected. Please check your dataset structure.")
        print("\nRequired structure:")
        print("Dataset/")
        print("├── Cat/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("└── Dog/")
        print("    ├── image1.jpg")
        print("    ├── image2.jpg")
        print("    └── ...")

if __name__ == "__main__":
    main()
