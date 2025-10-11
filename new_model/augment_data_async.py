# augment_data_async.py - ULTRA FAST async version

import asyncio
import aiohttp
from aiohttp import ClientSession
import pandas as pd
from tqdm.auto import tqdm
import json
from pathlib import Path
import time

class AsyncWikipediaAugmenter:
    """Ultra-fast async Wikipedia augmentation"""
    
    def __init__(self, cache_file='wiki_cache.json', max_concurrent=50):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    def load_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    async def fetch_wikipedia(self, session, item_name):
        """Fetch Wikipedia data async"""
        cache_key = item_name.lower()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = {'summary': '', 'found': False, 'languages': 0}
        
        try:
            async with self.semaphore:
                # Use Wikipedia API
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{item_name.replace(' ', '_')}"
                
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        result['found'] = True
                        result['summary'] = data.get('extract', '')[:500]
                        
                        # Get language count
                        lang_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={item_name}&prop=langlinks&lllimit=500&format=json"
                        async with session.get(lang_url, timeout=5) as lang_response:
                            if lang_response.status == 200:
                                lang_data = await lang_response.json()
                                pages = lang_data.get('query', {}).get('pages', {})
                                for page in pages.values():
                                    result['languages'] = len(page.get('langlinks', []))
        
        except Exception as e:
            pass
        
        self.cache[cache_key] = result
        return result
    
    async def augment_row_async(self, session, row):
        """Augment single row async"""
        wiki_data = await self.fetch_wikipedia(session, row['name'])
        
        enriched_parts = [row['description']]
        
        if wiki_data['summary']:
            enriched_parts.append(f"Wikipedia: {wiki_data['summary']}")
        
        if wiki_data['languages'] > 50:
            enriched_parts.append("Available in 50+ languages.")
        elif wiki_data['languages'] > 20:
            enriched_parts.append(f"Available in {wiki_data['languages']} languages.")
        
        augmented_row = row.copy()
        augmented_row['description'] = ' '.join(enriched_parts)
        augmented_row['wiki_found'] = wiki_data['found']
        augmented_row['wiki_languages'] = wiki_data['languages']
        
        return augmented_row
    
    async def augment_dataframe_async(self, df):
        """Augment entire dataframe async"""
        async with ClientSession() as session:
            tasks = []
            for _, row in df.iterrows():
                task = self.augment_row_async(session, row)
                tasks.append(task)
            
            results = []
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Augmenting"):
                result = await f
                results.append(result)
        
        self.save_cache()
        return pd.DataFrame(results)
    
    def augment_dataframe(self, df):
        """Sync wrapper for async augmentation"""
        return asyncio.run(self.augment_dataframe_async(df))

def main():
    print("🚀 Ultra-Fast Async Wikipedia Augmentation")
    
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    augmenter = AsyncWikipediaAugmenter(max_concurrent=50)
    
    print(f"\n📊 Training data: {len(train_df)} rows")
    start = time.time()
    train_aug = augmenter.augment_dataframe(train_df)
    print(f"⏱️  Took: {(time.time()-start)/60:.1f} min ({len(train_df)/(time.time()-start):.1f} rows/sec)")
    train_aug.to_csv("train_augmented.csv", index=False)
    
    print(f"\n📊 Test data: {len(test_df)} rows")
    start = time.time()
    test_aug = augmenter.augment_dataframe(test_df)
    print(f"⏱️  Took: {time.time()-start:.1f} sec")
    test_aug.to_csv("test_augmented.csv", index=False)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()