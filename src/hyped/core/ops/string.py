"""This module defines a collection of data processors that implement common string operations.

Each processor is designed to handle a specific string transformation or query, such as finding
substrings, replacing patterns, splitting strings, or trimming characters. These processors are
intended for use in data processing pipelines, where they can be applied in a batched and efficient
manner using Apache Arrow as the backend.

These processors are registered as methods on the :class:`StringFeature` class, allowing them to
be applied directly to string features. The operations are batched and optimized for high
performance using Arrow, making them suitable for large-scale data processing tasks.
"""

import pyarrow.compute as pc

from ..features.features import BoolFeature, Int32Feature, StringFeature
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Int, Sequence, String, UInt


class StringAddConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringAdd` processor."""


class StringAdd(BaseDataProcessor[StringAddConfig]):
    """Data processor for string concatenation."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, a: String, b: String) -> StringFeature:
        """Concatenates two strings element-wise.

        Args:
            ctx (RunContext): The execution context.
            a (String): The first string column.
            b (String): The second string column.

        Returns:
            StringFeature: The concatenated string column.
        """
        return pc.binary_join_element_wise(a, b, "")


class StringMultiplyConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringMultiply` processor."""


class StringMultiply(BaseDataProcessor[StringMultiplyConfig]):
    """Data processor for string multiplication."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, a: String, b: Int | UInt) -> StringFeature:
        """Repeats strings in column :code:`a` according to values in column :code:`b`.

        Args:
            ctx (RunContext): The execution context.
            a (String): The string column.
            b (Int | UInt): The column specifying the number of repetitions.

        Returns:
            StringFeature: The resulting column with repeated strings.
        """
        return pc.binary_repeat(a, b)


class StringFormatConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringFormat` processor."""


class StringFormat(BaseDataProcessor[StringFormatConfig]):
    """Data processor for string formatting."""

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, string: String, **kwargs: String) -> StringFeature:
        """Formats strings in the :code:`string` column using keyword arguments.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to format.
            **kwargs (String): Keyword arguments for formatting.

        Returns:
            StringFeature: The formatted string column.
        """
        return string.format(**kwargs)


class StringGetSliceConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringGetSlice` processor."""

    start: int
    """The starting index of the slice."""

    stop: None | int = None
    """The ending index of the slice."""

    step: int = 1
    """The step size for slicing (default is 1)."""


class StringGetSlice(BaseDataProcessor[StringGetSliceConfig]):
    """Data processor for slicing strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Extracts slices of strings according to the configuration.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to slice.

        Returns:
            StringFeature: The sliced string column.
        """
        return pc.utf8_slice_codeunits(
            string, self.config.start, self.config.stop, self.config.step
        )


class StringSetSliceConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringSetSlice` processor."""

    start: int
    """The starting index of the slice to replace."""

    stop: int
    """The ending index of the slice to replace."""

    replacement: str
    """The replacement string."""


class StringSetSlice(BaseDataProcessor[StringSetSliceConfig]):
    """Data processor for setting slices of strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Replaces slices of strings with the specified replacement string.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to modify.

        Returns:
            StringFeature: The modified string column with replaced slices.
        """
        return pc.utf8_replace_slice(
            string, self.config.start, self.config.stop, self.config.replacement
        )


class StringLengthConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringLength` processor."""


class StringLength(BaseDataProcessor[StringLengthConfig]):
    """Data processor for calculating the length of strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> Int32Feature:
        """Calculates the length of the input string column.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to calculate lengths for.

        Returns:
            Int32Feature: A column of integers representing the length of each string.
        """
        return pc.utf8_length(string)


class StringUpperConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringUpper` processor."""


class StringUpper(BaseDataProcessor[StringUpperConfig]):
    """Data processor for converting strings to uppercase."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Converts the input string column to uppercase.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to convert to uppercase.

        Returns:
            StringFeature: A column of uppercase strings.
        """
        return pc.utf8_upper(string)


class StringLowerConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringLower` processor."""


class StringLower(BaseDataProcessor[StringLowerConfig]):
    """Data processor for converting strings to lowercase."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Converts the input string column to lowercase.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to convert to lowercase.

        Returns:
            StringFeature: A column of lowercase strings.
        """
        return pc.utf8_lower(string)


class StringCapitalizeConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringCapitalize` processor."""


class StringCapitalize(BaseDataProcessor[StringCapitalizeConfig]):
    """Data processor for capitalizing strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Capitalizes the first letter of each string in the column.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to capitalize.

        Returns:
            StringFeature: A column of strings with the first letter capitalized.
        """
        return pc.utf8_capitalize(string)


class StringTitleConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringTitle` processor."""


class StringTitle(BaseDataProcessor[StringTitleConfig]):
    """Data processor for converting strings to title case."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Converts the input string column to title case.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to convert to title case.

        Returns:
            StringFeature: A column of strings in title case.
        """
        return pc.utf8_title(string)


class StringSwapCaseConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringSwapCase` processor."""


class StringSwapCase(BaseDataProcessor[StringSwapCaseConfig]):
    """Data processor for swapping the case of strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Swaps the case of each character in the input string column.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to swap the case for.

        Returns:
            StringFeature: A column with swapped case strings.
        """
        return pc.utf8_swapcase(string)


class StringStartsWithConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringStartsWith` processor."""

    pattern: str
    """The pattern to check for at the start of each string."""


class StringStartsWith(BaseDataProcessor[StringStartsWithConfig]):
    """Data processor for checking if strings start with a given pattern."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> BoolFeature:
        """Checks if the input string column starts with the specified pattern.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to check.

        Returns:
            BoolFeature: A column of boolean values indicating whether each string starts
            with the pattern.
        """
        return pc.starts_with(string, self.config.pattern)


class StringEndsWithConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringEndsWith` processor."""

    pattern: str
    """The pattern to check for at the end of each string."""


class StringEndsWith(BaseDataProcessor[StringEndsWithConfig]):
    """Data processor for checking if strings end with a given pattern."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> BoolFeature:
        """Checks if the input string column ends with the specified pattern.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to check.

        Returns:
            BoolFeature: A column of boolean values indicating whether each string ends
            with the pattern.
        """
        return pc.ends_with(string, self.config.pattern)


class StringReplaceConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringReplace` processor."""

    pattern: str
    """The pattern to search for in each string."""

    replacement: str
    """The string to replace the pattern with."""

    count: None | int = None
    """The maximum number of replacements to make (default is no limit)."""


class StringReplace(BaseDataProcessor[StringReplaceConfig]):
    """Data processor for replacing substrings in strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Replaces occurrences of the pattern with the replacement string.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to perform replacements on.

        Returns:
            StringFeature: A column of strings with the specified replacements.
        """
        return pc.replace_substring(
            string, self.config.pattern, self.config.replacement, max_replacements=self.config.count
        )


class StringFindConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringFind` processor."""

    pattern: str
    """The pattern to search for in each string."""


class StringFind(BaseDataProcessor[StringFindConfig]):
    """Data processor for finding the position of a pattern in strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> Int32Feature:
        """Finds the position of the specified pattern in each string.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to search.

        Returns:
            Int32Feature: A column of integer values representing the position of the pattern
            in each string.
        """
        return pc.find_substring(string, self.config.pattern)


class StringSplitConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringSplit` processor."""

    pattern: str
    """The pattern to split each string on."""

    max_splits: None | int = None
    """The maximum number of splits to perform (default is no limit)."""

    reverse: bool = False
    """Whether to perform the splits in reverse order."""


class StringSplit(BaseDataProcessor[StringSplitConfig]):
    """Data processor for splitting strings by a pattern."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> Sequence[StringFeature]:
        """Splits each string in the column by the specified pattern.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to split.

        Returns:
            Sequence[StringFeature]: A column of string sequences, resulting from the splits.
        """
        return pc.split_pattern(
            string,
            self.config.pattern,
            max_splits=self.config.max_splits,
            reverse=self.config.reverse,
        )


class StringStripConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringStrip` processor."""

    characters: str
    """The set of characters to remove from both ends of each string."""


class StringStrip(BaseDataProcessor[StringStripConfig]):
    """Data processor for stripping characters from both ends of strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Removes specified characters from both ends of the input string column.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to strip.

        Returns:
            StringFeature: A column of strings with the specified characters removed from both
            ends.
        """
        return pc.utf8_trim(string, self.config.characters)


class StringRightStripConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringRightStrip` processor."""

    characters: str
    """The set of characters to remove from the right end of each string."""


class StringRightStrip(BaseDataProcessor[StringRightStripConfig]):
    """Data processor for stripping characters from the right end of strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Removes specified characters from the right end of the input string column.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to right-strip.

        Returns:
            StringFeature: A column of strings with the specified characters removed from the right
            end.
        """
        return pc.utf8_rtrim(string, self.config.characters)


class StringLeftStripConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`StringLeftStrip` processor."""

    characters: str
    """The set of characters to remove from the left end of each string."""


class StringLeftStrip(BaseDataProcessor[StringLeftStripConfig]):
    """Data processor for stripping characters from the left end of strings."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, string: String) -> StringFeature:
        """Removes specified characters from the left end of the input string column.

        Args:
            ctx (RunContext): The execution context.
            string (String): The string column to left-strip.

        Returns:
            StringFeature: A column of strings with the specified characters removed from the left
            end.
        """
        return pc.utf8_ltrim(string, self.config.characters)


StringFeature.register_method("__add__")(
    StringFeature.register_method("__radd__")(StringAdd().call)
)
StringFeature.register_method("__mul__")(
    StringFeature.register_method("__rmul__")(StringMultiply().call)
)
StringFeature.register_method("length")(StringLength().call)
StringFeature.register_method("upper")(StringUpper().call)
StringFeature.register_method("lower")(StringLower().call)
StringFeature.register_method("capitalize")(StringCapitalize().call)
StringFeature.register_method("title")(StringTitle().call)
StringFeature.register_method("swapcase")(StringSwapCase().call)


@StringFeature.register_method("startswith")
def string_startswith(string: String, pattern: str) -> BoolFeature:
    """Checks if each string in the column starts with the specified pattern.

    Args:
        string (String): The string column to check.
        pattern (str): The pattern to check for at the start of each string.

    Returns:
        BoolFeature: A column of boolean values indicating whether each string starts with the
        pattern.
    """
    return StringStartsWith(pattern=pattern).call(string)


@StringFeature.register_method("endswith")
def string_endswith(string: String, pattern: str) -> BoolFeature:
    """Checks if each string in the column ends with the specified pattern.

    Args:
        string (String): The string column to check.
        pattern (str): The pattern to check for at the end of each string.

    Returns:
        BoolFeature: A column of boolean values indicating whether each string ends with the
        pattern.
    """
    return StringEndsWith(pattern=pattern).call(string)


@StringFeature.register_method("replace")
def string_replace(
    string: String, pattern: str, replacement: str, count: None | int = None
) -> StringFeature:
    """Replaces occurrences of the specified pattern in each string with the replacement string.

    Args:
        string (String): The string column to modify.
        pattern (str): The pattern to replace.
        replacement (str): The string to replace the pattern with.
        count (None | int, optional): The maximum number of occurrences to replace. If None, all
            occurrences are replaced.

    Returns:
        StringFeature: A column of strings with the specified pattern replaced by the replacement
        string.
    """
    return StringReplace(pattern=pattern, replacement=replacement, count=count).call(string)


@StringFeature.register_method("find")
def string_find(string: String, pattern: str) -> Int32Feature:
    """Finds the position of the specified pattern in each string.

    Args:
        string (String): The string column to search.
        pattern (str): The pattern to find in each string.

    Returns:
        Int32Feature: A column of integer values representing the position of the pattern in each
        string.
    """
    return StringFind(pattern=pattern).call(string)


@StringFeature.register_method("split")
def string_split(
    string: String, pattern: str, max_splits: None | int = None, reverse: bool = False
) -> StringFeature:
    """Splits each string in the column by the specified pattern.

    Args:
        string (String): The string column to split.
        pattern (str): The pattern to split each string on.
        max_splits (None | int, optional): The maximum number of splits to perform. If None, no
            limit is applied.
        reverse (bool, optional): Whether to split in reverse order.

    Returns:
        StringFeature: A column of string sequences resulting from splitting each string.
    """
    return StringSplit(pattern=pattern, max_splits=max_splits, reverse=reverse).call(string)


@StringFeature.register_method("strip")
def string_strip(string: String, characters: str) -> StringFeature:
    """Removes specified characters from both ends of each string.

    Args:
        string (String): The string column to strip.
        characters (str): The set of characters to remove from both ends of each string.

    Returns:
        StringFeature: A column of strings with the specified characters removed from both ends.
    """
    return StringStrip(characters=characters).call(string)


@StringFeature.register_method("rstrip")
def string_rstrip(string: String, characters: str) -> StringFeature:
    """Removes specified characters from the right end of each string.

    Args:
        string (String): The string column to right-strip.
        characters (str): The set of characters to remove from the right end of each string.

    Returns:
        StringFeature: A column of strings with the specified characters removed from the right end.
    """
    return StringRightStrip(characters=characters).call(string)


@StringFeature.register_method("lstrip")
def string_lstrip(string: String, characters: str) -> StringFeature:
    """Removes specified characters from the left end of each string.

    Args:
        string (String): The string column to left-strip.
        characters (str): The set of characters to remove from the left end of each string.

    Returns:
        StringFeature: A column of strings with the specified characters removed from the left end.
    """
    return StringLeftStrip(characters=characters).call(string)


@StringFeature.register_method("format")
def string_format(string: String, **kwargs: String) -> StringFeature:
    """Formats each string in the column using the provided keyword arguments.

    Args:
        string (String): The string column to format.
        **kwargs (String): The keyword arguments to substitute into the string.

    Returns:
        StringFeature: A column of formatted strings.
    """
    # TODO: support positional arguments
    return StringFormat().call(string, **kwargs)


@StringFeature.register_method("__getitem__")
def string_getitem(string: String, idx: int | slice) -> StringFeature:
    """Gets a slice of each string in the column using the specified index or slice.

    Args:
        string (String): The string column to slice.
        idx (int | slice): The index or slice specifying the range of characters to retrieve.

    Returns:
        StringFeature: A column of strings representing the sliced portion.
    """
    idx = slice(idx, idx + 1) if isinstance(idx, int) else idx
    idx = slice(idx.start or 0, idx.stop, idx.step or 1)
    node = StringGetSlice(start=idx.start, stop=idx.stop, step=idx.step)
    return node.call(string)


@StringFeature.register_method("__setitem__")
def string_setitem(string: String, idx: int | slice, replacement: str) -> StringFeature:
    """Sets a slice of each string in the column to the specified replacement string.

    Args:
        string (String): The string column to modify.
        idx (int | slice): The index or slice specifying the range of characters to replace.
        replacement (str): The string to replace the sliced portion with.

    Returns:
        StringFeature: A column of strings with the specified portion replaced by the replacement
        string.

    Raises:
        ValueError: If the slice step is not 1.
        ValueError: If the length of the replacement string does not match the slice range.
    """
    idx = slice(idx, idx + 1) if isinstance(idx, int) else idx
    idx = slice(idx.start or 0, idx.stop, idx.step or 1)

    if idx.step != 1:
        raise ValueError(f"Only step of 1 is supported for slice indexing, got {idx.step}.")

    if idx.stop - idx.start != len(replacement):
        raise ValueError(
            f"The length of the replacement string ({len(replacement)}) must match the length "
            f"of the slice range ({idx.stop - idx.start})."
        )

    node = StringSetSlice(start=idx.start, stop=idx.stop, replacement=replacement)
    return node.call(string)
