# check_augmentation.py - Quick statistics checker

import pandas as pd

def check_augmentation():
    print("="*80)
    print("ä AUGMENTATION STATISTICS")
    print("="*80)
    
    # Load files
    train_orig = pd.read_csv("train.csv")
    train_aug = pd.read_csv("train_augmented.csv")
    test_orig = pd.read_csv("test.csv")
    test_aug = pd.read_csv("test_augmented.csv")
    
    print("\nà Training Data:")
    print(f"  Rows: {len(train_aug)}")
    print(f"  Avg description length:")
    print(f"    Before: {train_orig['description'].str.len().mean():.0f} chars")
    print(f"    After:  {train_aug['description'].str.len().mean():.0f} chars")
    
    if 'wiki_found' in train_aug.columns:
        found = train_aug['wiki_found'].sum()
        print(f"  Wikipedia data found: {found}/{len(train_aug)} ({found/len(train_aug)*100:.1f}%)")
    
    print("\nà Test Data:")
    print(f"  Rows: {len(test_aug)}")
    print(f"  Avg description length:")
    print(f"    Before: {test_orig['description'].str.len().mean():.0f} chars")
    print(f"    After:  {test_aug['description'].str.len().mean():.0f} chars")
    
    if 'wiki_found' in test_aug.columns:
        found = test_aug['wiki_found'].sum()
        print(f"  Wikipedia data found: {found}/{len(test_aug)} ({found/len(test_aug)*100:.1f}%)")
    
    # Sample comparison
    print("\n" + "="*80)
    print("ù SAMPLE COMPARISON")
    print("="*80)
    
    idx = 0
    print(f"\nItem: {train_orig.iloc[idx]['name']}")
    print(f"\nBEFORE:")
    print(f"  {train_orig.iloc[idx]['description']}")
    print(f"\nAFTER:")
    print(f"  {train_aug.iloc[idx]['description'][:500]}...")

if __name__ == "__main__":
    check_augmentation()