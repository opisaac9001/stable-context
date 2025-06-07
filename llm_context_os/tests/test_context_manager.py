# yawl/tests/test_context_manager.py
import unittest
from yawl.context.context_manager import ContextManager, MockTokenizer # Updated import

# --- Conditional imports for HFTokenizer ---
HAVE_TRANSFORMERS = False
HF_TOKENIZER_INSTANCE = None
HF_MODEL_NAME = "gpt2" # Using a common model, fairly small

try:
    # Try to import AutoTokenizer to confirm transformers library is available
    from transformers import AutoTokenizer
    from yawl.context.token_estimator import HFTokenEstimator # Updated import
    HAVE_TRANSFORMERS = True
    try:
        HF_TOKENIZER_INSTANCE = HFTokenEstimator(model_name=HF_MODEL_NAME)
        print(f"Successfully loaded HFTokenizer with model '{HF_MODEL_NAME}'.")
    except Exception as e:
        # This can happen if the model is not found, network issues, etc.
        print(f"Note: Could not load HFTokenizer model '{HF_MODEL_NAME}': {e}")
        HF_TOKENIZER_INSTANCE = None # Ensure it's None if model loading fails
except ImportError:
    print("Note: 'transformers' library not found or HFTokenEstimator not found. Skipping HFTokenizer tests.")
    # HAVE_TRANSFORMERS remains False, HF_TOKENIZER_INSTANCE remains None
    pass
# --- End conditional imports ---

class TestContextManager(unittest.TestCase):

    def setUp(self):
        self.mock_tokenizer = MockTokenizer() # Using the char-counting mock
        self.system_prompt_str = "SYSTEM:" # Renamed from self.system_prompt to avoid conflict if HF has own
        # For MockTokenizer (char-based):
        # System prompt "SYSTEM:" is 7 chars. Newline is 1 char. So 8 chars for system part.

    def test_initialization(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        self.assertEqual(cm.system_prompt, self.system_prompt_str)
        self.assertEqual(cm._messages, [])
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.max_tokens, 100)
        # Initial prompt should just be the system prompt.
        # build_prompt returns a list of dicts, the first being system if present.
        expected_prompt_list = [{'role': 'system', 'content': self.system_prompt_str}]
        self.assertEqual(cm.build_prompt(), expected_prompt_list)


    def test_add_message_simple(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "Hello") # image_path defaults to None
        self.assertEqual(len(cm._messages), 1)
        # Check the actual message content, excluding None image_path if it's not set
        expected_message = {"r": "user", "c": "Hello"}
        # self.assertEqual(cm._messages[0], {"r": "user", "c": "Hello", "image_path": None})
        self.assertDictContainsSubset(expected_message, cm._messages[0])


        self.assertEqual(cm._start, 0)
        # build_prompt returns list of dicts
        expected_prompt_list = [
            {'role': 'system', 'content': self.system_prompt_str},
            {'role': 'user', 'content': 'Hello'}
        ]
        self.assertEqual(cm.build_prompt(), expected_prompt_list)

    def test_add_multiple_messages_and_auto_scroll(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "Msg1")
        cm.add("ai", "Msg2")
        self.assertEqual(len(cm._messages), 2)
        self.assertEqual(cm._start, 1) # _auto_scroll sets _start to len - 1 then _fit

        # _fit logic:
        # Current window message: {'r': 'ai', 'c': 'Msg2'}
        # _get_prompt_string_for_fitting with _start=1: "SYSTEM:\nai: Msg2\n"
        # MockTokenizer counts: 7+1+9 = 17 chars. Fits in 100. _start remains 1.
        expected_prompt_list = [
            {'role': 'system', 'content': self.system_prompt_str},
            {'role': 'ai', 'content': 'Msg2'}
        ]
        self.assertEqual(cm.build_prompt(), expected_prompt_list)

    def _run_fit_logic_truncation_test(self, tokenizer, system_prompt_str, max_tokens_val, msg_contents, expected_final_start_idx, test_label=""):
        cm = ContextManager(system_prompt_str, tokenizer, max_tokens=max_tokens_val)
        roles = ["u", "a", "u"]
        for i in range(3):
            cm.add(roles[i], msg_contents[i])
            self.assertEqual(cm._start, i, f"[{test_label}] After {i+1} add(s), _start should be {i}")
            current_prompt_str_for_counting = cm._get_prompt_string_for_fitting() # Test with string version for count
            self.assertTrue(tokenizer.count_tokens(current_prompt_str_for_counting) <= max_tokens_val,
                            f"[{test_label}] Prompt after {i+1} add(s) should fit. Got {tokenizer.count_tokens(current_prompt_str_for_counting)}, max {max_tokens_val}. Prompt str: '{current_prompt_str_for_counting}'")

        cm.jump_to(0)
        final_prompt_str_for_counting = cm._get_prompt_string_for_fitting()
        final_tokens = tokenizer.count_tokens(final_prompt_str_for_counting)
        self.assertEqual(cm._start, expected_final_start_idx,
                         f"[{test_label}] After jump_to(0) and _fit, _start should be {expected_final_start_idx}, but got {cm._start}. "
                         f"Final prompt str: '{final_prompt_str_for_counting}', tokens: {final_tokens}, max_tokens: {max_tokens_val}")
        self.assertTrue(final_tokens <= max_tokens_val,
                        f"[{test_label}] Final prompt tokens ({final_tokens}) should not exceed max_tokens ({max_tokens_val}). "
                        f"Prompt str: '{final_prompt_str_for_counting}'")
        if expected_final_start_idx > 0 and len(cm._messages) > 0:
            first_message_content = cm._messages[0]['c']
            # Check if first_message_content appears in the final_prompt_str_for_counting
            self.assertNotIn(f"u: {first_message_content}", final_prompt_str_for_counting) # Assuming role 'u' for first message
            self.assertNotIn(f"a: {first_message_content}", final_prompt_str_for_counting) # Assuming role 'a' for first message


    def test_fit_logic_truncation(self):
        system_prompt = self.system_prompt_str # "SYSTEM:" (7 chars)
        # For _get_prompt_string_for_fitting: "SYSTEM:\n" (8)
        # Message "user: content01\n": 6 + 9 + 1 = 16 chars (includes "user: " and newline)
        # Message "ai: content01\n": 4 + 9 + 1 = 14 chars
        msg_contents = ["content01", "content02", "content03"] # 9 chars each

        # Case 1: String "SYSTEM:\nuser: content03\n" -> 8 + 16 = 24. Max 30.
        self._run_fit_logic_truncation_test(self.mock_tokenizer, system_prompt,
                                            max_tokens_val=30, msg_contents=msg_contents,
                                            expected_final_start_idx=2, test_label="Mock-TruncateToLast")

        # Case 2: String "SYSTEM:\nuser: content01\nai: content02\nuser: content03\n" -> 8 + 16 + 14 + 16 = 54. Max 55.
        self._run_fit_logic_truncation_test(self.mock_tokenizer, system_prompt,
                                            max_tokens_val=55, msg_contents=msg_contents,
                                            expected_final_start_idx=0, test_label="Mock-AllFit")

        # Case 3: String "SYSTEM:\nai: content02\nuser: content03\n" -> 8 + 14 + 16 = 38. Max 40.
        self._run_fit_logic_truncation_test(self.mock_tokenizer, system_prompt,
                                            max_tokens_val=40, msg_contents=msg_contents,
                                            expected_final_start_idx=1, test_label="Mock-TruncateToOne")


    @unittest.skipIf(not HF_TOKENIZER_INSTANCE, "HFTokenizer not available or model could not be loaded")
    def test_fit_logic_truncation_hf(self):
        system_prompt = self.system_prompt_str # "SYSTEM:" (gpt2: 2 tokens)
        # For _get_prompt_string_for_fitting: "SYSTEM:\n" (3 tokens)
        # Message "user: content01\n" (gpt2: "user"(1) + ":"(1) + " content"(1) + "01"(1) + "\n"(1) = 5 tokens)
        msg_contents = ["content01", "content02", "content03"]

        # Total tokens if all included: Sys(3) + M1(5) + M2(5) + M3(5) = 18 tokens
        self._run_fit_logic_truncation_test(HF_TOKENIZER_INSTANCE, system_prompt,
                                            max_tokens_val=20, msg_contents=msg_contents,
                                            expected_final_start_idx=0, test_label="HF-AllFit")
        # Sys(3) + M2(5) + M3(5) = 13. Max 15.
        self._run_fit_logic_truncation_test(HF_TOKENIZER_INSTANCE, system_prompt,
                                            max_tokens_val=15, msg_contents=msg_contents,
                                            expected_final_start_idx=1, test_label="HF-TruncateToOne")
        # Sys(3) + M3(5) = 8. Max 9.
        self._run_fit_logic_truncation_test(HF_TOKENIZER_INSTANCE, system_prompt,
                                            max_tokens_val=9, msg_contents=msg_contents,
                                            expected_final_start_idx=2, test_label="HF-TruncateToLast")

    def test_system_prompt_never_dropped(self):
        # System: 7. Max tokens 10. Budget for messages (including role prefix and NL): 10 - 7 = 3
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=10)
        # "user: 12345\n" is 6+5+1 = 12 chars. Prompt "SYSTEM:\nuser: 12345\n" (7+1+12 = 20 tokens). Too long.
        cm.add("user", "12345")
        self.assertEqual(cm._start, 0)
        expected_prompt_list = [{'role':'system', 'content':self.system_prompt_str}, {'role':'user', 'content':'12345'}]
        self.assertEqual(cm.build_prompt(), expected_prompt_list)
        self.assertTrue(self.mock_tokenizer.count_tokens(cm._get_prompt_string_for_fitting()) > cm.max_tokens)

        cm.add("ai", "1") # "ai: 1\n" (6 chars)
        # _auto_scroll -> _start=1.
        # _fit window=[m2]: "SYSTEM:\nai: 1\n" (7+1+6=14 chars) > 10.
        # _start remains 1 as it cannot be incremented further.
        self.assertEqual(cm._start, 1)
        expected_prompt_list_2 = [{'role':'system', 'content':self.system_prompt_str}, {'role':'ai', 'content':'1'}]
        self.assertEqual(cm.build_prompt(), expected_prompt_list_2)


    def test_shift_navigation(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100) # Increased max_tokens
        # System: "SYSTEM:\n" (8 chars)
        # "user: msg1\n" (6+4+1=11), "ai: msg2\n" (4+4+1=9), "user: msg3\n" (11), "ai: msg4\n" (9)
        cm.add("user", "msg1"); cm.add("ai", "msg2"); cm.add("user", "msg3"); cm.add("ai", "msg4")
        self.assertEqual(cm._start, 3) # Points to msg4
        self.assertEqual(cm.build_prompt(), [{'role':'system', 'content':self.system_prompt_str}, {'role':'ai', 'content':'msg4'}])

        cm.shift(-1); self.assertEqual(cm._start, 2) # Points to msg3. Window: [msg3, msg4]
        expected_list1 = [{'role':'system', 'content':self.system_prompt_str}, {'role':'user', 'content':'msg3'}, {'role':'ai', 'content':'msg4'}]
        self.assertEqual(cm.build_prompt(), expected_list1)

        cm.shift(-2); self.assertEqual(cm._start, 0) # Points to msg1. Window: [msg1, msg2, msg3, msg4]
        expected_list2 = [{'role':'system', 'content':self.system_prompt_str}, {'role':'user', 'content':'msg1'}, {'role':'ai', 'content':'msg2'}, {'role':'user', 'content':'msg3'}, {'role':'ai', 'content':'msg4'}]
        self.assertEqual(cm.build_prompt(), expected_list2)

        cm.shift(1); self.assertEqual(cm._start, 1) # Points to msg2. Window: [msg2, msg3, msg4]
        expected_list3 = [{'role':'system', 'content':self.system_prompt_str}, {'role':'ai', 'content':'msg2'}, {'role':'user', 'content':'msg3'}, {'role':'ai', 'content':'msg4'}]
        self.assertEqual(cm.build_prompt(), expected_list3)

        cm.shift(10); self.assertEqual(cm._start, 3)
        cm.shift(-20); self.assertEqual(cm._start, 0)


    def test_jump_to_navigation(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "msg1"); cm.add("ai", "msg2"); cm.add("user", "msg3"); cm.add("ai", "msg4")

        cm.jump_to(0); self.assertEqual(cm._start, 0)
        expected_list1 = [{'role':'system', 'content':self.system_prompt_str}, {'role':'user', 'content':'msg1'}, {'role':'ai', 'content':'msg2'}, {'role':'user', 'content':'msg3'}, {'role':'ai', 'content':'msg4'}]
        self.assertEqual(cm.build_prompt(), expected_list1)

        cm.jump_to(2); self.assertEqual(cm._start, 2)
        expected_list2 = [{'role':'system', 'content':self.system_prompt_str}, {'role':'user', 'content':'msg3'}, {'role':'ai', 'content':'msg4'}]
        self.assertEqual(cm.build_prompt(), expected_list2)

        cm.jump_to(3); self.assertEqual(cm._start, 3)
        expected_list3 = [{'role':'system', 'content':self.system_prompt_str},{'role':'ai', 'content':'msg4'}]
        self.assertEqual(cm.build_prompt(), expected_list3)

        cm.jump_to(10); self.assertEqual(cm._start, 3)
        cm.jump_to(-5); self.assertEqual(cm._start, 0)

    def test_empty_messages_navigation(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=20)
        self.assertEqual(cm.build_prompt(), [{'role':'system', 'content':self.system_prompt_str}])
        cm.shift(1); self.assertEqual(cm._start, 0)
        cm.shift(-1); self.assertEqual(cm._start, 0)
        cm.jump_to(0); self.assertEqual(cm._start, 0)
        cm.jump_to(5); self.assertEqual(cm._start, 0)


    def test_build_prompt_with_extra_messages(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "msg1")
        extra_msgs = [{"r": "system", "c": "hint"}] # In internal format

        # Expected: System, then extra (if system role, might merge or appear after), then window
        # Current build_prompt puts system first, then extras, then window.
        # If extra is 'system', it results in two system messages if main system_prompt also exists.
        # This might be model-dependent on how it's handled.
        # For this test, let's assume extra messages are non-system or are appended.
        extra_msgs_user_role = [{"r": "user", "c": "This is a RAG snippet."}]
        prompt_list = cm.build_prompt(extra_messages=extra_msgs_user_role)

        expected_list = [
            {'role': 'system', 'content': self.system_prompt_str},
            {'role': 'user', 'content': 'This is a RAG snippet.'}, # Extra message
            {'role': 'user', 'content': 'msg1'} # Window message
        ]
        self.assertEqual(prompt_list, expected_list)
        self.assertEqual(len(cm._messages), 1); self.assertEqual(cm._start, 0) # Original state unchanged
        self.assertEqual(cm.build_prompt(), [{'role':'system', 'content':self.system_prompt_str}, {'role':'user', 'content':'msg1'}])

    def test_build_prompt_with_extra_messages_causing_overflow_no_fit_on_build(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=20)
        cm.add("user", "msg1")
        extra_msgs = [{"r": "user", "c": "A very very long RAG snippet that will surely cause an overflow if counted."}]
        prompt_list = cm.build_prompt(extra_messages=extra_msgs)

        # Check the string representation for token counting (this is what _fit uses)
        prompt_str_for_counting = cm._format_messages_for_token_counting(prompt_list) # Pass the full list
        self.assertTrue(self.mock_tokenizer.count_tokens(prompt_str_for_counting) > cm.max_tokens)


    def test_fit_with_single_very_long_message(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=10)
        cm.add("user", "12345") # This message + system prompt will exceed max_tokens
        self.assertEqual(cm._start, 0) # _fit doesn't truncate if only one message beyond system
        expected_list = [{'role':'system', 'content':self.system_prompt_str}, {'role':'user', 'content':'12345'}]
        self.assertEqual(cm.build_prompt(), expected_list)
        self.assertTrue(self.mock_tokenizer.count_tokens(cm._get_prompt_string_for_fitting()) > cm.max_tokens)


    def test_add_until_first_message_is_pushed_out(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=35) # Max for Sys + 2 msgs
        # Sys (8) + msg1 (u:item_A\n -> 6+6+1=13) + msg2 (a:item_B\n -> 4+6+1=11) = 8+13+11 = 32. Fits.
        # Sys (8) + msg2 (11) + msg3 (u:item_C\n -> 13) = 8+11+13 = 32. Fits.
        cm.add("user", "item_A"); self.assertEqual(cm._start, 0)
        cm.add("ai", "item_B"); self.assertEqual(cm._start, 1)
        cm.add("user", "item_C"); self.assertEqual(cm._start, 2) # Current window is just item_C + system.

        cm.jump_to(0) # Attempt to show all: Sys + M1 + M2 + M3 (8 + 13 + 11 + 13 = 45) > 35
        # _fit runs:
        # start=0, prompt=45 > 35. start becomes 1.
        # start=1, prompt=Sys+M2+M3 = 8+11+13=32 <= 35. Loop ends. _start=1.
        self.assertEqual(cm._start, 1)
        expected_list = [{'role':'system', 'content':self.system_prompt_str}, {'role':'ai', 'content':'item_B'}, {'role':'user', 'content':'item_C'}]
        self.assertEqual(cm.build_prompt(), expected_list)

    def test_add_and_format_with_image_path(self):
        cm = ContextManager("SysPrompt:", self.mock_tokenizer, max_tokens=200)
        cm.add(role="user", content="Look at this picture of a cat.", image_path="path/to/cat.jpg")
        cm.add(role="user", content="And this one of a dog.", image_path="path/to/dog.png")
        cm.add(role="assistant", content="Interesting pictures!")
        cm.add(role="user", content="What was the first picture about?")
        cm.jump_to(0)
        prompt_list = cm.build_prompt()
        # Check structure for the first user message with image
        self.assertEqual(prompt_list[1]['role'], 'user')
        self.assertEqual(prompt_list[1]['content'], "Look at this picture of a cat.")
        self.assertEqual(prompt_list[1]['image_path_for_runner_handling'], "path/to/cat.jpg")

        # Test _get_prompt_string_for_fitting includes image placeholder
        prompt_str_for_fitting = cm._get_prompt_string_for_fitting()
        self.assertIn("[IMAGE: path/to/cat.jpg] Look at this picture of a cat.", prompt_str_for_fitting)
        self.assertIn("[IMAGE: path/to/dog.png] And this one of a dog.", prompt_str_for_fitting)

        cm.max_tokens = 150
        cm.jump_to(0)
        self.assertEqual(cm._start, 1) # Based on char counts from previous version of this test
        prompt_after_fit_list = cm.build_prompt()
        prompt_after_fit_str = cm._get_prompt_string_for_fitting() # Get string for checking content
        self.assertNotIn("[IMAGE: path/to/cat.jpg]", prompt_after_fit_str)
        self.assertIn("[IMAGE: path/to/dog.png]", prompt_after_fit_str)

    def assertDictContainsSubset(self, subset, main_dict): # Helper for comparing dicts ignoring None values from one side
        for key, value in subset.items():
            self.assertIn(key, main_dict)
            self.assertEqual(main_dict[key], value)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

[end of llm_context_os/tests/test_context_manager.py]

[start of llm_context_os/tests/test_kv_cache_manager.py]
# llm_context_os/tests/test_kv_cache_manager.py
import unittest
import shutil
from pathlib import Path
import typing as t
import pickle # For dummy file content in new test

from llm_context_os.caching.kv_cache_manager import KVCacheManager

class TestKVCacheManager(unittest.TestCase):
    def setUp(self):
        self.test_data_dir_root = Path("data_test_kv_cache_manager_module") # Main test dir for this class
        self.test_cache_dir = self.test_data_dir_root / "kv_cache_storage" # Cache storage for KVCacheManager
        self.source_files_temp_dir = self.test_data_dir_root / "temp_source_files" # For creating dummy source files

        # Clean up and create fresh directories for each test method
        if self.test_data_dir_root.exists():
            shutil.rmtree(self.test_data_dir_root)
        self.test_cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_files_temp_dir.mkdir(parents=True, exist_ok=True)

        self.kv_manager = KVCacheManager(cache_dir=str(self.test_cache_dir))

        self.mock_model_id = "test_model/v1.0_gguf"
        self.mock_prefix_text = "This is a test prefix that will be hashed."
        self.mock_pickle_data = [
            {'key_cache_layer_0': [[0.1, 0.2, 0.3]], 'value_cache_layer_0': [[0.4, 0.5, 0.6]]},
            {'key_cache_layer_1': [[0.7, 0.8]], 'value_cache_layer_1': [[0.9, 1.0]]}
        ]

        # Create a dummy source file for file copy tests
        self.dummy_source_file_path = self.source_files_temp_dir / "source_session.kst"
        self.dummy_source_file_content = {"session_data": "this is a dummy KST file content", "version": 1}
        with open(self.dummy_source_file_path, 'wb') as f:
            pickle.dump(self.dummy_source_file_content, f) # Using pickle for dummy content

    def tearDown(self):
        if self.test_data_dir_root.exists():
            shutil.rmtree(self.test_data_dir_root)

    def test_instantiation_creates_directory(self):
        temp_dir_for_inst_test = self.test_data_dir_root / "temp_instantiation_test_cache"
        if temp_dir_for_inst_test.exists(): # Should not exist due to root cleanup in setUp
            shutil.rmtree(temp_dir_for_inst_test)
        self.assertFalse(temp_dir_for_inst_test.exists())
        KVCacheManager(cache_dir=str(temp_dir_for_inst_test)) # This should create it
        self.assertTrue(temp_dir_for_inst_test.exists())

    def test_save_and_load_pickle_data(self): # Renamed
        # Test saving pickled data
        save_success = self.kv_manager.save_kv_cache(
            self.mock_pickle_data, self.mock_model_id, self.mock_prefix_text
        )
        self.assertTrue(save_success, "Failed to save KV cache (pickled data).")

        prefix_hash = self.kv_manager._get_prefix_hash(self.mock_prefix_text)
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        expected_filename = f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl" # Default pickle extension
        expected_filepath = self.test_cache_dir / expected_filename
        self.assertTrue(expected_filepath.exists(), f"Pickled cache file was not created at {expected_filepath}.")

        # Test loading pickled data - raw_file_type_suffix should be None or omitted for pickled objects
        loaded_data = self.kv_manager.load_kv_cache(self.mock_model_id, self.mock_prefix_text, raw_file_type_suffix=None)
        self.assertIsNotNone(loaded_data, "Failed to load KV cache (pickled data).")
        self.assertEqual(loaded_data, self.mock_pickle_data, "Loaded pickled data does not match saved data.")

    def test_save_and_load_file_copy_data(self):
        # Test saving a file (by copying)
        save_success = self.kv_manager.save_kv_cache(
            str(self.dummy_source_file_path), self.mock_model_id, self.mock_prefix_text
        )
        self.assertTrue(save_success, "Failed to save KV cache (file copy).")

        prefix_hash = self.kv_manager._get_prefix_hash(self.mock_prefix_text)
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        # Expected filename in cache dir will have the original suffix
        expected_cached_filename = f"{sanitized_model_id}_{prefix_hash}{self.dummy_source_file_path.suffix}"
        expected_cached_filepath = self.test_cache_dir / expected_cached_filename
        self.assertTrue(expected_cached_filepath.exists(), f"Copied cache file was not created at {expected_cached_filepath}.")

        # Verify content of the copied file
        with open(expected_cached_filepath, 'rb') as f:
            copied_content = pickle.load(f) # We know original dummy was pickled
        self.assertEqual(copied_content, self.dummy_source_file_content, "Content of copied cache file differs from source.")

        # Test loading the path to the copied file
        loaded_path_str = self.kv_manager.load_kv_cache(
            self.mock_model_id, self.mock_prefix_text, raw_file_type_suffix=self.dummy_source_file_path.suffix
        )
        self.assertIsNotNone(loaded_path_str, "Failed to load KV cache (file copy path).")
        self.assertIsInstance(loaded_path_str, str, "load_kv_cache should return a string path for file copies.")
        self.assertEqual(Path(loaded_path_str), expected_cached_filepath.resolve(), "Loaded path does not match expected cached file path.")

        # Verify content again from loaded_path_str
        with open(loaded_path_str, 'rb') as f:
            content_from_loaded_path = pickle.load(f)
        self.assertEqual(content_from_loaded_path, self.dummy_source_file_content)


    def test_load_priority_file_copy_over_pickle(self):
        # 1. Save data via pickling for a specific model/prefix
        self.kv_manager.save_kv_cache(self.mock_pickle_data, self.mock_model_id, self.mock_prefix_text)

        # 2. Save a file copy for the *same* model/prefix (KVCacheManager should allow this, creating a different filename due to suffix)
        # This assumes KVCacheManager differentiates by suffix, which it does.
        # The save_kv_cache for file copy will create modelid_hash.kst
        # The save_kv_cache for pickle will create modelid_hash.kvcache.pkl
        self.kv_manager.save_kv_cache(str(self.dummy_source_file_path), self.mock_model_id, self.mock_prefix_text)

        # 3. Call load_kv_cache expecting the file copy path (.kst)
        loaded_path = self.kv_manager.load_kv_cache(self.mock_model_id, self.mock_prefix_text, raw_file_type_suffix=".kst")
        self.assertIsInstance(loaded_path, str, "Should have loaded the path to the .kst file.")
        with open(loaded_path, 'rb') as f:
            copied_content = pickle.load(f)
        self.assertEqual(copied_content, self.dummy_source_file_content)

        # 4. Call load_kv_cache expecting the pickled data (by passing raw_file_type_suffix=None or omitting it)
        loaded_object = self.kv_manager.load_kv_cache(self.mock_model_id, self.mock_prefix_text)
        self.assertEqual(loaded_object, self.mock_pickle_data, "Should have loaded the pickled object.")


    def test_load_non_existent_cache(self):
        # Test with raw_file_type_suffix=None (or omitted) for a non-existent pickled object
        loaded_data = self.kv_manager.load_kv_cache(self.mock_model_id, "a_completely_non_existent_prefix_text")
        self.assertIsNone(loaded_data, "Should return None for non-existent cache (pickled object).")

        # Test with a specific raw_file_type_suffix for a non-existent raw file
        loaded_path = self.kv_manager.load_kv_cache(self.mock_model_id, "another_non_existent_prefix", raw_file_type_suffix=".kst")
        self.assertIsNone(loaded_path, "Should return None for non-existent .kst cache.")


    def test_save_none_data(self):
        save_success = self.kv_manager.save_kv_cache(None, self.mock_model_id, "prefix_for_none_data_test")
        self.assertFalse(save_success, "Should return False when trying to save None data.")
        # Verify no file was created (neither .pkl nor any other type)
        prefix_hash = self.kv_manager._get_prefix_hash("prefix_for_none_data_test")
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        expected_pkl_filepath = self.test_cache_dir / f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl"
        self.assertFalse(expected_pkl_filepath.exists(), "Pickle cache file should not have been created for None data.")
        # Check for a common copied file extension too just in case
        expected_kst_filepath = self.test_cache_dir / f"{sanitized_model_id}_{prefix_hash}.kst"
        self.assertFalse(expected_kst_filepath.exists(), "Copied cache file should not have been created for None data.")


    def test_sanitization_in_filename(self):
        problematic_model_id = "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
        # Test with pickling
        self.kv_manager.save_kv_cache(self.mock_pickle_data, problematic_model_id, "prefix_pickle")
        prefix_hash_pickle = self.kv_manager._get_prefix_hash("prefix_pickle")
        sanitized_id = self.kv_manager._sanitize_identifier(problematic_model_id)
        expected_pickle_path = self.test_cache_dir / f"{sanitized_id}_{prefix_hash_pickle}.kvcache.pkl"
        self.assertTrue(expected_pickle_path.exists())

        # Test with file copy
        self.kv_manager.save_kv_cache(str(self.dummy_source_file_path), problematic_model_id, "prefix_filecopy")
        prefix_hash_filecopy = self.kv_manager._get_prefix_hash("prefix_filecopy")
        expected_filecopy_path = self.test_cache_dir / f"{sanitized_id}_{prefix_hash_filecopy}{self.dummy_source_file_path.suffix}"
        self.assertTrue(expected_filecopy_path.exists())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

[end of llm_context_os/tests/test_kv_cache_manager.py]

[start of llm_context_os/tests/test_token_estimator.py]
# llm_context_os/tests/test_token_estimator.py
import unittest
import typing as t

from llm_context_os.context.token_estimator import (
    BaseTokenEstimator,
    CharTokenEstimator,
    HFTokenEstimator,
    LlamaCppTokenEstimator,
    EXL2TokenEstimator,
    count_tokens
)

# --- Conditional imports for HFTokenizer ---
HAVE_TRANSFORMERS = False
HF_GPT2_TOKENIZER_INSTANCE: t.Optional[HFTokenEstimator] = None
_HF_MODEL_NAME_FOR_TEST = "gpt2" # Using a common model

try:
    from transformers import AutoTokenizer # To check if lib is there AND for direct use in tests
    _hf_auto_tokenizer_gpt2_for_test_comparison = AutoTokenizer.from_pretrained(_HF_MODEL_NAME_FOR_TEST)
    HAVE_TRANSFORMERS = True
    try:
        HF_GPT2_TOKENIZER_INSTANCE = HFTokenEstimator(model_name=_HF_MODEL_NAME_FOR_TEST)
        print(f"Successfully loaded HFTokenEstimator with model '{_HF_MODEL_NAME_FOR_TEST}' for tests.")
    except Exception as e:
        print(f"Note: Could not load HFTokenEstimator model '{_HF_MODEL_NAME_FOR_TEST}' for tests: {e}")
        HF_GPT2_TOKENIZER_INSTANCE = None
except ImportError:
    print("Note: 'transformers' library not found. Skipping some HFTokenEstimator tests.")
    AutoTokenizer = None # Ensure it's defined for type hinting or conditional checks if needed later
    _hf_auto_tokenizer_gpt2_for_test_comparison = None
except Exception as e: # Catch potential errors from AutoTokenizer.from_pretrained directly
    print(f"Note: Could not load AutoTokenizer for '{_HF_MODEL_NAME_FOR_TEST}' for tests: {e}")
    AutoTokenizer = None
    _hf_auto_tokenizer_gpt2_for_test_comparison = None
    HF_GPT2_TOKENIZER_INSTANCE = None # Ensure this is also None
# --- End conditional imports ---


class TestCharTokenEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = CharTokenEstimator()

    def test_count_tokens_empty_string(self):
        self.assertEqual(self.estimator.count_tokens(""), 0)

    def test_count_tokens_simple_string(self):
        self.assertEqual(self.estimator.count_tokens("hello"), 5)

    def test_count_tokens_with_spaces(self):
        self.assertEqual(self.estimator.count_tokens(" hello world "), 13)

    def test_count_tokens_none(self):
        self.assertEqual(self.estimator.count_tokens(None), 0)


class TestHFTokenEstimator(unittest.TestCase):
    def setUp(self):
        if not HF_GPT2_TOKENIZER_INSTANCE:
            self.skipTest(f"Hugging Face tokenizer '{_HF_MODEL_NAME_FOR_TEST}' not available for tests.")
        self.hf_tokenizer_instance = HF_GPT2_TOKENIZER_INSTANCE
        # For direct comparison if needed and AutoTokenizer loaded successfully
        self.gpt2_auto_tokenizer = _hf_auto_tokenizer_gpt2_for_test_comparison


    def test_count_tokens_empty_string(self):
        self.assertEqual(self.hf_tokenizer_instance.count_tokens(""), 0)

    def test_count_tokens_simple_sentence(self):
        text = "Hello, world!"
        # Expected for gpt2: "Hello" (1), "," (1), " world" (1), "!" (1) -> 4 tokens
        # Or using AutoTokenizer directly to get expected count:
        if self.gpt2_auto_tokenizer:
            expected_count = len(self.gpt2_auto_tokenizer.encode(text))
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), expected_count)
        else: # Fallback if direct AutoTokenizer failed but instance somehow loaded
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), 4)


    def test_count_tokens_common_words(self):
        text = "This is a simple sentence."
        if self.gpt2_auto_tokenizer:
            expected_count = len(self.gpt2_auto_tokenizer.encode(text))
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), expected_count)
        else:
            # "This" (1) " is" (1) " a" (1) " simple" (1) " sentence" (1) "." (1) -> 6 tokens for gpt2
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), 6)

    def test_count_tokens_none(self):
        self.assertEqual(self.hf_tokenizer_instance.count_tokens(None), 0)

    @unittest.skipIf(not HAVE_TRANSFORMERS, "Transformers library not available for this test.")
    def test_invalid_model_name(self):
        with self.assertRaisesRegex(ValueError, "Failed to load tokenizer for model 'invalid-model-name-does-not-exist'"):
            HFTokenEstimator("invalid-model-name-does-not-exist")

    # Testing ImportError if transformers is not installed is hard when it IS installed.
    # This would typically be done by manipulating sys.modules or in a separate test environment.
    # The HFTokenEstimator's __init__ checks `if AutoTokenizer is None`, which is covered
    # by the global setup if transformers import fails. An explicit test for that path:
    @unittest.skipIf(HAVE_TRANSFORMERS, "Skipping test for ImportError as transformers IS installed.")
    def test_hf_estimator_import_error_if_no_transformers(self):
        # This test will only run if the initial import of AutoTokenizer failed.
        with self.assertRaisesRegex(ImportError, "transformers library is not installed"):
            HFTokenEstimator("gpt2")


class TestPlaceholderTokenEstimators(unittest.TestCase):
    def test_llama_cpp_not_implemented(self):
        estimator = LlamaCppTokenEstimator("dummy/path/model.gguf")
        with self.assertRaisesRegex(NotImplementedError, "LlamaCppTokenEstimator is not yet implemented"):
            estimator.count_tokens("test text")

    def test_exl2_not_implemented(self):
        estimator = EXL2TokenEstimator("dummy/path/exl2_model")
        with self.assertRaisesRegex(NotImplementedError, "EXL2TokenEstimator is not yet implemented"):
            estimator.count_tokens("test text")


class TestCountTokensFunction(unittest.TestCase):
    def test_char_method(self):
        self.assertEqual(count_tokens("hello", method='char'), 5)

    @unittest.skipIf(not HF_GPT2_TOKENIZER_INSTANCE, f"HF tokenizer '{_HF_MODEL_NAME_FOR_TEST}' not available.")
    def test_hf_method_valid(self):
        text = "Hello, world!"
        if _hf_auto_tokenizer_gpt2_for_test_comparison:
            expected_count = len(_hf_auto_tokenizer_gpt2_for_test_comparison.encode(text))
            self.assertEqual(count_tokens(text, method='hf', model_name=_HF_MODEL_NAME_FOR_TEST), expected_count)
        else:
             self.assertEqual(count_tokens(text, method='hf', model_name=_HF_MODEL_NAME_FOR_TEST), 4) # Fallback


    @unittest.skipIf(not HAVE_TRANSFORMERS, "Transformers library not available for this test.")
    def test_hf_method_invalid_model(self):
        with self.assertRaisesRegex(ValueError, "Error using 'hf' method: Failed to load tokenizer"):
            count_tokens("test", method='hf', model_name="invalid-model-name-for-sure")

    def test_hf_method_missing_model_name(self):
        with self.assertRaisesRegex(ValueError, "model_name must be provided for 'hf' method"):
            count_tokens("test", method='hf')

    def test_llama_cpp_method_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "LlamaCppTokenEstimator is not yet implemented"):
            count_tokens("test", method='llama_cpp', model_path="dummy/path.gguf")

    def test_llama_cpp_method_missing_model_path(self):
        with self.assertRaisesRegex(ValueError, "model_path must be provided for 'llama_cpp' method"):
            count_tokens("test", method='llama_cpp')

    def test_exl2_method_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "EXL2TokenEstimator is not yet implemented"):
            count_tokens("test", method='exl2', model_dir="dummy/exl2_dir")

    def test_exl2_method_missing_model_dir(self):
        with self.assertRaisesRegex(ValueError, "model_dir must be provided for 'exl2' method"):
            count_tokens("test", method='exl2')

    def test_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown token estimation method: unknown_method"):
            count_tokens("test", method='unknown_method')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

[end of llm_context_os/tests/test_token_estimator.py]
