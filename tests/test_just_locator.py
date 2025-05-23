"""
Unit tests for the JustLocator implementation.
"""

import unittest
import gc
import threading
import time
import random
import asyncio
from typing import List, Optional
from dataclasses import dataclass
import sys
import os

# Add the core module to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from just_agents.just_locator import JustLocator, EntityIdentifier
from just_agents.just_async import run_async_function_synchronously


class MockEntity:
    """Mock entity class with name attribute."""
    
    def __init__(self, name: str, value: int = 0):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"MockEntity(name='{self.name}', value={self.value})"
    
    def __eq__(self, other):
        if not isinstance(other, MockEntity):
            return False
        return self.name == other.name and self.value == other.value


class MockEntityNoName:
    """Mock entity class without name attribute."""
    
    def __init__(self, data: str):
        self.data = data
    
    def __repr__(self):
        return f"MockEntityNoName(data='{self.data}')"


class MockEntityWithTitle:
    """Mock entity class with title attribute."""
    
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
    
    def __repr__(self):
        return f"MockEntityWithTitle(title='{self.title}', description='{self.description}')"


@dataclass
class ThreadTestEntity:
    """Simple test entity for demonstrating thread safety."""
    name: str
    value: int
    
    def __str__(self) -> str:
        return f"ThreadTestEntity(name='{self.name}', value={self.value})"


class ThreadSafetyWorker:
    """Worker class for performing thread-safe operations on the locator."""
    
    def __init__(self, locator: JustLocator[ThreadTestEntity], thread_id: int, operations: int):
        self.locator = locator
        self.thread_id = thread_id
        self.operations = operations
        self.results = []
        self.entities_created = []
        self.error_count = 0
    
    def run(self) -> List[str]:
        """Execute random operations on the locator."""
        for i in range(self.operations):
            operation = random.choice(['publish', 'lookup', 'unpublish', 'list_all'])
            
            try:
                if operation == 'publish':
                    self._publish_entity(i)
                elif operation == 'lookup' and self.entities_created:
                    self._lookup_entity()
                elif operation == 'unpublish' and self.entities_created:
                    self._unpublish_entity()
                elif operation == 'list_all':
                    self._list_all_entities()
                    
                # Small random delay to increase chance of race conditions if they exist
                time.sleep(random.uniform(0.001, 0.01))
                
            except Exception as e:
                self.error_count += 1
                self.results.append(f"Thread {self.thread_id}: ERROR in {operation}: {e}")
        
        # Clean up remaining entities
        self._cleanup_entities()
        return self.results
    
    def _publish_entity(self, index: int):
        """Publish a new entity."""
        entity = ThreadTestEntity(name=f"thread_{self.thread_id}_entity_{index}", 
                          value=random.randint(1, 100))
        codename = self.locator.publish_entity(entity)
        self.entities_created.append((entity, codename))
        self.results.append(f"Thread {self.thread_id}: Published {entity} with codename {codename}")
    
    def _lookup_entity(self):
        """Look up a random entity we created."""
        entity, codename = random.choice(self.entities_created)
        found_entity = self.locator.get_entity_by_codename(codename)
        
        if found_entity is entity:
            self.results.append(f"Thread {self.thread_id}: Successfully found {entity}")
        else:
            self.error_count += 1
            self.results.append(f"Thread {self.thread_id}: LOOKUP MISMATCH for {codename}")
    
    def _unpublish_entity(self):
        """Unpublish a random entity we created."""
        entity, codename = self.entities_created.pop(random.randint(0, len(self.entities_created) - 1))
        success = self.locator.unpublish_entity(entity)
        self.results.append(f"Thread {self.thread_id}: Unpublished {entity}, success: {success}")
        
        if not success:
            self.error_count += 1
    
    def _list_all_entities(self):
        """List all entities."""
        all_entities = self.locator.get_all_entities()
        self.results.append(f"Thread {self.thread_id}: Found {len(all_entities)} total entities")
    
    def _cleanup_entities(self):
        """Clean up any remaining entities."""
        for entity, codename in self.entities_created:
            try:
                success = self.locator.unpublish_entity(entity)
                self.results.append(f"Thread {self.thread_id}: Cleaned up {entity}, success: {success}")
                if not success:
                    self.error_count += 1
            except Exception as e:
                self.error_count += 1
                self.results.append(f"Thread {self.thread_id}: ERROR cleaning up {entity}: {e}")


class TestJustLocator(unittest.TestCase):
    """Test cases for JustLocator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.locator = JustLocator[MockEntity](entity_config_identifier_attr="name")
        
    def test_publish_entity_basic(self):
        """Test basic entity publishing."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        self.assertIsInstance(codename, str)
        self.assertGreater(len(codename), 0)
        
        # Verify entity can be retrieved
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIs(retrieved, entity)
        
    def test_publish_same_entity_twice(self):
        """Test that publishing the same entity twice returns the same codename."""
        entity = MockEntity("test", 42)
        codename1 = self.locator.publish_entity(entity)
        codename2 = self.locator.publish_entity(entity)
        
        self.assertEqual(codename1, codename2)
        
    def test_get_entity_codename(self):
        """Test getting codename for an entity instance."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        retrieved_codename = self.locator.get_entity_codename(entity)
        self.assertEqual(retrieved_codename, codename)
        
    def test_get_entity_codename_unregistered(self):
        """Test getting codename for unregistered entity."""
        entity = MockEntity("test", 42)
        codename = self.locator.get_entity_codename(entity)
        self.assertIsNone(codename)
        
    def test_get_entities_by_config_identifier(self):
        """Test getting entities by config identifier."""
        entity1 = MockEntity("alice", 1)
        entity2 = MockEntity("bob", 2)
        entity3 = MockEntity("alice", 3)
        
        self.locator.publish_entity(entity1)
        self.locator.publish_entity(entity2)
        self.locator.publish_entity(entity3)
        
        alices = self.locator.get_entities_by_config_identifier("alice")
        self.assertEqual(len(alices), 2)
        self.assertIn(entity1, alices)
        self.assertIn(entity3, alices)
        
        bobs = self.locator.get_entities_by_config_identifier("bob")
        self.assertEqual(len(bobs), 1)
        self.assertIn(entity2, bobs)
        
    def test_get_entity_codenames_by_config_identifier(self):
        """Test getting codenames by config identifier."""
        entity1 = MockEntity("alice", 1)
        entity2 = MockEntity("alice", 2)
        
        codename1 = self.locator.publish_entity(entity1)
        codename2 = self.locator.publish_entity(entity2)
        
        codenames = self.locator.get_entity_codenames_by_config_identifier("alice")
        self.assertEqual(len(codenames), 2)
        self.assertIn(codename1, codenames)
        self.assertIn(codename2, codenames)
        
    def test_get_entity_codenames_by_class(self):
        """Test getting codenames by class."""
        entity1 = MockEntity("test1", 1)
        entity2 = MockEntity("test2", 2)
        
        codename1 = self.locator.publish_entity(entity1)
        codename2 = self.locator.publish_entity(entity2)
        
        codenames = self.locator.get_entity_codenames_by_class(MockEntity)
        self.assertEqual(len(codenames), 2)
        self.assertIn(codename1, codenames)
        self.assertIn(codename2, codenames)
        
    def test_arbitrary_search(self):
        """Test arbitrary search functionality."""
        entity1 = MockEntity("test1", 10)
        entity2 = MockEntity("test2", 5)
        entity3 = MockEntity("test3", 15)
        
        self.locator.publish_entity(entity1)
        self.locator.publish_entity(entity2)
        self.locator.publish_entity(entity3)
        
        # Find entities with value > 7
        results = self.locator.arbitrary_search(MockEntity, lambda e: e.value > 7)
        self.assertEqual(len(results), 2)
        self.assertIn(entity1, results)
        self.assertIn(entity3, results)
        
    def test_unpublish_entity(self):
        """Test unpublishing an entity."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        # Verify entity exists
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIs(retrieved, entity)
        
        # Unpublish
        result = self.locator.unpublish_entity(entity)
        self.assertTrue(result)
        
        # Verify entity is gone
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIsNone(retrieved)
        
    def test_unpublish_entity_by_codename(self):
        """Test unpublishing by codename."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        result = self.locator.unpublish_entity_by_codename(codename)
        self.assertTrue(result)
        
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIsNone(retrieved)
        
    def test_unpublish_entities_by_config_identifier(self):
        """Test unpublishing by config identifier."""
        entity1 = MockEntity("alice", 1)
        entity2 = MockEntity("alice", 2)
        entity3 = MockEntity("bob", 3)
        
        self.locator.publish_entity(entity1)
        self.locator.publish_entity(entity2)
        self.locator.publish_entity(entity3)
        
        removed_count = self.locator.unpublish_entities_by_config_identifier("alice")
        self.assertEqual(removed_count, 2)
        
        # Verify alice entities are gone
        alices = self.locator.get_entities_by_config_identifier("alice")
        self.assertEqual(len(alices), 0)
        
        # Verify bob entity still exists
        bobs = self.locator.get_entities_by_config_identifier("bob")
        self.assertEqual(len(bobs), 1)
        
    def test_get_identifier_by_instance(self):
        """Test getting identifier by entity instance."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        identifier = self.locator.get_identifier_by_instance(entity)
        self.assertIsNotNone(identifier)
        self.assertEqual(identifier.entity_codename, codename)
        self.assertEqual(identifier.entity_class, MockEntity)
        self.assertEqual(identifier.entity_config_identifier, "test")


class TestEntityConfigIdentifierFallback(unittest.TestCase):
    """Test cases for entity config identifier fallback behavior."""
    
    def test_fallback_to_class_name(self):
        """Test fallback to class name when attribute doesn't exist."""
        locator = JustLocator[MockEntityNoName]()  # Uses default 'name' attr
        
        entity = MockEntityNoName("test_data")
        codename = locator.publish_entity(entity)
        
        # Should be able to find by class name since 'name' attribute doesn't exist
        entities = locator.get_entities_by_config_identifier("MockEntityNoName")
        self.assertEqual(len(entities), 1)
        self.assertIs(entities[0], entity)
        
    def test_custom_attribute_name(self):
        """Test using a custom attribute name for config identifier."""
        locator = JustLocator[MockEntityWithTitle](entity_config_identifier_attr="title")
        
        entity1 = MockEntityWithTitle("Product A", "Description A")
        entity2 = MockEntityWithTitle("Product B", "Description B")
        entity3 = MockEntityWithTitle("Product A", "Different description")
        
        locator.publish_entity(entity1)
        locator.publish_entity(entity2)
        locator.publish_entity(entity3)
        
        # Find by title
        product_a_entities = locator.get_entities_by_config_identifier("Product A")
        self.assertEqual(len(product_a_entities), 2)
        self.assertIn(entity1, product_a_entities)
        self.assertIn(entity3, product_a_entities)
        
        product_b_entities = locator.get_entities_by_config_identifier("Product B")
        self.assertEqual(len(product_b_entities), 1)
        self.assertIn(entity2, product_b_entities)


class TestEntityIdentifier(unittest.TestCase):
    """Test cases for EntityIdentifier."""
    
    def test_entity_identifier_creation(self):
        """Test creating an EntityIdentifier."""
        identifier = EntityIdentifier[MockEntity](
            entity_class=MockEntity,
            entity_codename="test-codename"
        )
        
        self.assertEqual(identifier.entity_class, MockEntity)
        self.assertEqual(identifier.entity_codename, "test-codename")
        
    def test_entity_config_identifier_property(self):
        """Test the entity_config_identifier property."""
        locator = JustLocator[MockEntity](entity_config_identifier_attr="name")
        entity = MockEntity("test_name", 42)
        
        codename = locator.publish_entity(entity)
        identifier = locator.get_identifier_by_instance(entity)
        
        self.assertEqual(identifier.entity_config_identifier, "test_name")
        
    def test_entity_config_identifier_without_locator(self):
        """Test that accessing entity_config_identifier without locator raises error."""
        identifier = EntityIdentifier[MockEntity](
            entity_class=MockEntity,
            entity_codename="test-codename"
        )
        
        with self.assertRaises(ValueError):
            _ = identifier.entity_config_identifier


class TestWeakReferences(unittest.TestCase):
    """Test cases for weak reference management."""
    
    def test_automatic_cleanup_on_garbage_collection(self):
        """Test that entities are automatically cleaned up when garbage collected."""
        locator = JustLocator[MockEntity](entity_config_identifier_attr="name")
        
        # Create and register an entity
        entity = MockEntity("test", 42)
        codename = locator.publish_entity(entity)
        
        # Verify entity exists
        retrieved = locator.get_entity_by_codename(codename)
        self.assertIs(retrieved, entity)
        
        # Remove our reference and force aggressive garbage collection
        del entity
        del retrieved  # Remove the retrieved reference too
        # Force multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()
        
        # Entity should be automatically cleaned up
        retrieved = locator.get_entity_by_codename(codename)
        self.assertIsNone(retrieved)
        
        # Identifier should also be cleaned up
        self.assertNotIn(codename, locator._entity_codename_to_identifiers)


class TestThreadSafety(unittest.TestCase):
    """Test cases for JustLocator thread safety."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.locator = JustLocator[ThreadTestEntity](entity_config_identifier_attr="name")
    
    def test_concurrent_operations_stress_test(self):
        """Test that the locator handles concurrent operations correctly."""
        print("\n🧵 Starting thread safety stress test...")
        
        # Configuration
        num_threads = 10
        operations_per_thread = 50
        
        # Create worker containers
        workers = []
        threads = []
        
        def thread_wrapper(worker: ThreadSafetyWorker):
            """Wrapper function for thread execution."""
            try:
                worker.run()
            except Exception as e:
                worker.error_count += 1
                worker.results.append(f"Thread {worker.thread_id} CRASHED: {e}")
        
        print(f"🚀 Starting {num_threads} threads with {operations_per_thread} operations each...")
        start_time = time.time()
        
        # Create and start threads
        for i in range(num_threads):
            worker = ThreadSafetyWorker(self.locator, i, operations_per_thread)
            workers.append(worker)
            thread = threading.Thread(target=thread_wrapper, args=(worker,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Analyze results
        total_operations = 0
        total_errors = 0
        
        print("\n📊 Results Summary:")
        print("=" * 50)
        
        for worker in workers:
            thread_operations = len(worker.results)
            thread_errors = worker.error_count
            
            total_operations += thread_operations
            total_errors += thread_errors
            
            print(f"Thread {worker.thread_id}: {thread_operations} operations, {thread_errors} errors")
            
            # Show first few results for each thread
            for result in worker.results[:3]:
                print(f"  • {result}")
            if len(worker.results) > 3:
                print(f"  • ... and {len(worker.results) - 3} more")
        
        print("=" * 50)
        print(f"✅ Total operations: {total_operations}")
        print(f"❌ Total errors: {total_errors}")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        
        final_entity_count = len(self.locator.get_all_entities())
        print(f"🏁 Final entity count: {final_entity_count}")
        
        # Assertions
        self.assertGreater(total_operations, 0, "No operations were performed")
        self.assertEqual(total_errors, 0, f"Thread safety violations detected: {total_errors} errors")
        self.assertEqual(final_entity_count, 0, "Memory leak: entities not properly cleaned up")
        
        print("\n🎉 SUCCESS: All operations completed without errors!")
        print("   The locator is confirmed to be thread-safe.")
    
    def test_async_compatibility(self):
        """Test that the thread-safe locator works with async code."""
        print("\n🔄 Testing async compatibility...")
        
        async def async_worker(worker_id: int) -> str:
            """Async worker that uses the locator."""
            entity = ThreadTestEntity(name=f"async_worker_{worker_id}", value=worker_id * 10)
            
            # These operations are thread-safe and work fine from async contexts
            codename = self.locator.publish_entity(entity)
            
            # Verify we can find it
            found_entity = self.locator.get_entity_by_codename(codename)
            self.assertIs(found_entity, entity, f"Failed to retrieve entity {codename}")
            
            # Clean up
            success = self.locator.unpublish_entity(entity)
            self.assertTrue(success, f"Failed to unpublish entity {codename}")
            
            return f"Async worker {worker_id}: published, found, unpublished entity successfully"
        
        async def run_async_workers():
            """Run multiple async workers concurrently."""
            tasks = [async_worker(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results
        
        # Use the utility function to run async code synchronously
        try:
            results = run_async_function_synchronously(run_async_workers)
            
            # Assertions
            self.assertEqual(len(results), 5, "Not all async workers completed")
            for i, result in enumerate(results):
                self.assertIn(f"Async worker {i}:", result)
                self.assertIn("successfully", result)
            
            # Verify no entities left behind
            remaining_entities = self.locator.get_all_entities()
            self.assertEqual(len(remaining_entities), 0, "Async workers left entities behind")
            
            print("✅ Async compatibility test passed:")
            for result in results:
                print(f"  • {result}")
                
        except Exception as e:
            self.fail(f"❌ Async compatibility test failed: {e}")
    
    def test_mixed_threading_and_async(self):
        """Test mixing threaded and async operations."""
        print("\n🔄 Testing mixed threading and async operations...")
        
        def mixed_worker(worker_id: int) -> str:
            """Thread that runs async operations."""
            async def async_operations():
                entity = ThreadTestEntity(name=f"mixed_worker_{worker_id}", value=worker_id * 5)
                codename = self.locator.publish_entity(entity)
                
                # Verify retrieval
                found = self.locator.get_entity_by_codename(codename)
                if found is not entity:
                    raise ValueError(f"Failed to retrieve entity {codename}")
                
                # Clean up
                success = self.locator.unpublish_entity(entity)
                if not success:
                    raise ValueError(f"Failed to unpublish entity {codename}")
                
                return codename
            
            return run_async_function_synchronously(async_operations)
        
        # Run multiple threads, each running async code
        num_threads = 5
        results = []
        threads = []
        exceptions = []
        
        def thread_wrapper(worker_id: int):
            """Wrapper to catch exceptions."""
            try:
                result = mixed_worker(worker_id)
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Start threads
        for i in range(num_threads):
            thread = threading.Thread(target=thread_wrapper, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Assertions
        self.assertEqual(len(exceptions), 0, f"Mixed operations failed: {exceptions}")
        self.assertEqual(len(results), num_threads, "Not all mixed workers completed")
        
        # Verify no entities left behind
        remaining_entities = self.locator.get_all_entities()
        self.assertEqual(len(remaining_entities), 0, "Mixed workers left entities behind")
        
        print(f"✅ Mixed threading/async test passed: {num_threads} workers completed")
        print("   All operations successful with no race conditions detected")
    
    def test_high_contention_scenario(self):
        """Test a high-contention scenario with many threads accessing the same entities."""
        print("\n⚡ Testing high contention scenario...")
        
        # Pre-populate with some entities
        shared_entities = []
        for i in range(5):
            entity = ThreadTestEntity(name=f"shared_{i}", value=i * 10)
            codename = self.locator.publish_entity(entity)
            shared_entities.append((entity, codename))
        
        results = []
        errors = []
        
        def high_contention_worker(worker_id: int):
            """Worker that accesses shared entities frequently."""
            worker_results = []
            worker_errors = []
            
            for i in range(20):
                try:
                    # Random operation on shared entities
                    operation = random.choice(['lookup', 'list_by_name', 'list_all'])
                    
                    if operation == 'lookup':
                        entity, codename = random.choice(shared_entities)
                        found = self.locator.get_entity_by_codename(codename)
                        if found is not entity:
                            worker_errors.append(f"Worker {worker_id}: Lookup failed for {codename}")
                        else:
                            worker_results.append(f"Worker {worker_id}: Found {codename}")
                    
                    elif operation == 'list_by_name':
                        entity, _ = random.choice(shared_entities)
                        entities = self.locator.get_entities_by_config_identifier(entity.name)
                        if entity not in entities:
                            worker_errors.append(f"Worker {worker_id}: Entity {entity.name} not in list")
                        else:
                            worker_results.append(f"Worker {worker_id}: Listed {len(entities)} entities for {entity.name}")
                    
                    elif operation == 'list_all':
                        all_entities = self.locator.get_all_entities()
                        if len(all_entities) < len(shared_entities):
                            worker_errors.append(f"Worker {worker_id}: Missing entities in list_all")
                        else:
                            worker_results.append(f"Worker {worker_id}: Listed {len(all_entities)} total entities")
                    
                    time.sleep(random.uniform(0.001, 0.005))
                    
                except Exception as e:
                    worker_errors.append(f"Worker {worker_id}: Exception in {operation}: {e}")
            
            results.extend(worker_results)
            errors.extend(worker_errors)
        
        # Start multiple threads
        num_threads = 8
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(target=high_contention_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Clean up shared entities
        for entity, codename in shared_entities:
            success = self.locator.unpublish_entity(entity)
            self.assertTrue(success, f"Failed to clean up entity {codename}")
        
        # Assertions
        self.assertEqual(len(errors), 0, f"High contention test failed: {errors[:5]}")  # Show first 5 errors
        self.assertGreater(len(results), 0, "No operations completed in high contention test")
        
        final_count = len(self.locator.get_all_entities())
        self.assertEqual(final_count, 0, "Entities not properly cleaned up after high contention test")
        
        print(f"✅ High contention test passed: {len(results)} operations across {num_threads} threads")
        print("   No race conditions or data corruption detected")


if __name__ == '__main__':
    unittest.main() 