from hyped.core.ops.string import (
    StringAdd,
    StringCapitalize,
    StringContains,
    StringEndsWith,
    StringFind,
    StringFormat,
    StringGetSlice,
    StringLeftStrip,
    StringLength,
    StringLower,
    StringMultiply,
    StringReplace,
    StringRightStrip,
    StringSetSlice,
    StringSplit,
    StringStartsWith,
    StringStrip,
    StringSwapCase,
    StringTitle,
    StringUpper,
)
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.typing import Bool, Int, Int32, Sequence, String


class TestStringAdd(BaseDataProcessorTest):
    processor = StringAdd()
    input_features = {"a": String, "b": String}
    input_data = [{"a": "First", "b": "Second"}, {"a": "One", "b": "Two"}]
    expected_output_feature = String
    expected_output_data = ["FirstSecond", "OneTwo"]


class TestStringMultiply(BaseDataProcessorTest):
    processor = StringMultiply()
    input_features = {"a": String, "b": Int}
    input_data = [{"a": "Repeat", "b": 3}, {"a": "Test", "b": 2}]
    expected_output_feature = String
    expected_output_data = ["RepeatRepeatRepeat", "TestTest"]


class TestStringFormatBasic(BaseDataProcessorTest):
    processor = StringFormat()
    input_features = {"string": String, "name": String}
    input_data = [
        {"string": "Hello, {name}!", "name": "Alice"},
        {"string": "Goodbye, {name}!", "name": "Bob"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "Hello, Alice!",
        "Goodbye, Bob!",
    ]


class TestStringGetSliceStartStop(BaseDataProcessorTest):
    processor = StringGetSlice(start=1, stop=4)
    input_features = {"string": String}
    input_data = [
        {"string": "Hello"},
        {"string": "World"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "ell",  # Slicing "Hello" from index 1 to 4
        "orl",  # Slicing "World" from index 1 to 4
    ]


class TestStringGetSliceStartStopStep(BaseDataProcessorTest):
    processor = StringGetSlice(start=0, stop=5, step=2)
    input_features = {"string": String}
    input_data = [
        {"string": "abcdef"},
        {"string": "123456"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "ace",  # Slicing "abcdef" from index 0 to 5 with a step of 2
        "135",  # Slicing "123456" from index 0 to 5 with a step of 2
    ]


class TestStringSetSliceBasic(BaseDataProcessorTest):
    processor = StringSetSlice(start=1, stop=4, replacement="XYZ")
    input_features = {"string": String}
    input_data = [
        {"string": "abcdef"},
        {"string": "123456"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "aXYZef",  # Replace indices 1 to 4 in "abcdef" with "XYZ"
        "1XYZ56",  # Replace indices 1 to 4 in "123456" with "XYZ"
    ]


class TestStringSetSliceFull(BaseDataProcessorTest):
    processor = StringSetSlice(start=0, stop=6, replacement="New")
    input_features = {"string": String}
    input_data = [
        {"string": "abcdef"},
        {"string": "123456"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "New",  # Replace the entire string "abcdef" with "New"
        "New",  # Replace the entire string "123456" with "New"
    ]


class TestStringLengthBasic(BaseDataProcessorTest):
    processor = StringLength()
    input_features = {"string": String}
    input_data = [
        {"string": "hello"},
        {"string": "world"},
    ]
    expected_output_feature = Int32
    expected_output_data = [
        5,  # Length of "hello"
        5,  # Length of "world"
    ]


class TestStringLengthSpecialCharacters(BaseDataProcessorTest):
    processor = StringLength()
    input_features = {"string": String}
    input_data = [
        {"string": "こんにちは"},  # Japanese (5 characters)
        {"string": "😊🌟💡"},  # Emoji (3 characters)
    ]
    expected_output_feature = Int32
    expected_output_data = [
        5,  # Length of "こんにちは"
        3,  # Length of "😊🌟💡"
    ]


class TestStringUpper(BaseDataProcessorTest):
    processor = StringUpper()
    input_features = {"string": String}
    input_data = [
        {"string": "hello"},
        {"string": "world"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "HELLO",
        "WORLD",
    ]


class TestStringLower(BaseDataProcessorTest):
    processor = StringLower()
    input_features = {"string": String}
    input_data = [
        {"string": "HELLO"},
        {"string": "WORLD"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "hello",
        "world",
    ]


class TestStringCapitalize(BaseDataProcessorTest):
    processor = StringCapitalize()
    input_features = {"string": String}
    input_data = [
        {"string": "hello"},
        {"string": "WORLD"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "Hello",
        "World",
    ]


class TestStringTitle(BaseDataProcessorTest):
    processor = StringTitle()
    input_features = {"string": String}
    input_data = [
        {"string": "hello world"},
        {"string": "PYTHON TEST"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "Hello World",
        "Python Test",
    ]


class TestStringSwapCase(BaseDataProcessorTest):
    processor = StringSwapCase()
    input_features = {"string": String}
    input_data = [
        {"string": "Hello World"},
        {"string": "PYTHON test"},
    ]
    expected_output_feature = String
    expected_output_data = [
        "hELLO wORLD",
        "python TEST",
    ]


class TestStringStartsWith(BaseDataProcessorTest):
    processor = StringStartsWith(pattern="Test")
    input_features = {"string": String}
    input_data = [
        {"string": "TestString"},
        {"string": "ExampleString"},
    ]
    expected_output_feature = Bool
    expected_output_data = [
        True,  # Starts with "Test"
        False,  # Does not start with "Test"
    ]


class TestStringEndsWith(BaseDataProcessorTest):
    processor = StringEndsWith(pattern="Test")
    input_features = {"string": String}
    input_data = [
        {"string": "UnitTest"},
        {"string": "IntegrationTest"},
        {"string": "Performance"},
    ]
    expected_output_feature = Bool
    expected_output_data = [
        True,  # Ends with "Test"
        True,  # Ends with "Test"
        False,  # Does not end with "Test"
    ]


class TestStringReplace(BaseDataProcessorTest):
    processor = StringReplace(pattern="foo", replacement="bar", count=1)
    input_features = {"string": String}
    input_data = [
        {"string": "foo foo foo"},
        {"string": "hello world"},
        {"string": ""},
    ]
    expected_output_feature = String
    expected_output_data = [
        "bar foo foo",  # Only first occurrence replaced
        "hello world",  # No match
        "",  # Empty input
    ]


class TestStringFind(BaseDataProcessorTest):
    processor = StringFind(pattern="test")
    input_features = {"string": String}
    input_data = [
        {"string": "this is a test"},
        {"string": "no match here"},
        {"string": ""},
    ]
    expected_output_feature = Int32
    expected_output_data = [
        10,  # Position of "test"
        -1,  # No match
        -1,  # Empty input
    ]


class TestStringContains(BaseDataProcessorTest):
    processor = StringContains(pattern="test")
    input_features = {"string": String}
    input_data = [
        {"string": "this is a test"},
        {"string": "no match here"},
        {"string": ""},
    ]
    expected_output_feature = Bool
    expected_output_data = [
        True,
        False,
        False,
    ]


class TestStringSplit(BaseDataProcessorTest):
    processor = StringSplit(pattern=",", max_splits=2, reverse=False)
    input_features = {"string": String}
    input_data = [
        {"string": "a,b,c,d"},
        {"string": "no commas here"},
        {"string": ""},
    ]
    expected_output_feature = Sequence[String]
    expected_output_data = [
        ["a", "b", "c,d"],  # Split on 2 commas
        ["no commas here"],  # No split
        [""],  # Empty input
    ]


class TestStringStrip(BaseDataProcessorTest):
    processor = StringStrip(characters=" *")
    input_features = {"string": String}
    input_data = [
        {"string": "  hello world  "},
        {"string": "**hello**"},
        {"string": "no-special-characters"},
        {"string": ""},
    ]
    expected_output_feature = String
    expected_output_data = [
        "hello world",  # Strips spaces and asterisks
        "hello",  # Strips asterisks
        "no-special-characters",  # No matching characters to strip
        "",  # Empty input remains empty
    ]


class TestStringRightStrip(BaseDataProcessorTest):
    processor = StringRightStrip(characters=" *")
    input_features = {"string": String}
    input_data = [
        {"string": "  hello world  "},
        {"string": "**hello**"},
        {"string": "no-special-characters"},
        {"string": ""},
    ]
    expected_output_feature = String
    expected_output_data = [
        "  hello world",  # Strips spaces from the right
        "**hello",  # Strips trailing asterisks
        "no-special-characters",  # No matching characters to strip
        "",  # Empty input remains empty
    ]


class TestStringLeftStrip(BaseDataProcessorTest):
    processor = StringLeftStrip(characters=" *")
    input_features = {"string": String}
    input_data = [
        {"string": "  hello world  "},
        {"string": "**hello**"},
        {"string": "no-special-characters"},
        {"string": ""},
    ]
    expected_output_feature = String
    expected_output_data = [
        "hello world  ",  # Strips leading spaces
        "hello**",  # Strips leading asterisks
        "no-special-characters",  # No matching characters to strip
        "",  # Empty input remains empty
    ]
