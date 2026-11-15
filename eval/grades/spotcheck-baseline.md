# Spot-check: baseline answer grades

Grades: `eval/grades/20261014T172908-152daa3-answers-mlx.yaml` (model-assisted, pending human spot-check).
Run: `eval/results/20261014T172908-152daa3-answers-mlx.json` (plain prompt, mlx, dense+graph retrieval).

All 40 verdicts: correct 20, partial 9, correctly abstained 7, wrong 4.

Below: the 12 flagged questions, then 3 others chosen with `random.Random(20261115).sample(<the other 28 ids, sorted>, 3)`: h01, h14, q05.

Grounding is judged only against the lines shown to the model (see eval/grades/README.md). Excerpts show the shown
lines around what the grader cited; the full shown range is in each `[C#]` location.

---

## q11 (location, stdlib) — flagged

**Question:** Where does json notice that an object contains itself and refuse to encode it?

**Classification:** answered

**Answer:**

> The JSON encoder checks for circular references by using a markers dictionary during the encoding process. If `check_circular` is set to `True`, the encoder maintains a dictionary (`markers`) to track objects that have already been seen during encoding. When an object is encountered that is already in the markers dictionary, it indicates a circular reference, and the encoder will refuse to encode it to prevent infinite recursion. This behavior is implemented in the `iterencode` method of `JSONEncoder`, where the `markers` dictionary is initialized if `check_circular` is `True` [C2, C5].

**Relevant (eval set):** `json/encoder.py::_make_iterencode._iterencode_list`, `json/encoder.py::_make_iterencode._iterencode_dict`, `json/encoder.py::_make_iterencode._iterencode`

**Shown context:**
- [C1] `json/encoder.py::JSONEncoder.__init__` json/encoder.py:105-159
- [C2] `json/encoder.py::JSONEncoder` json/encoder.py:74-261 (cited)
- [C3] `json/__init__.py::dumps` json/__init__.py:183-238
- [C4] `json/__init__.py::<module>` json/__init__.py:1-102,104-104,106-108,110-118,241-241
- [C5] `json/encoder.py::JSONEncoder.iterencode` json/encoder.py:205-261 (cited)
- [C6] `json/decoder.py::JSONDecoder` json/decoder.py:262-364
- [C7] `json/encoder.py::JSONEncoder.encode` json/encoder.py:183-203
- [C8] `json/encoder.py::JSONEncoder.default` json/encoder.py:161-181

[C2] json/encoder.py (shown lines near the grader’s citations):
```python
  213  
  214          """
  215          if self.check_circular:
  216              markers = {}
  217          else:
  218              markers = None
  219          if self.ensure_ascii:
  220              _encoder = encode_basestring_ascii
```

**Verdict:** partial
**Justification:** supported: iterencode creates the markers dict only when check_circular is true (json/encoder.py:215-218), and the check exists to prevent infinite recursion (encoder.py:118-121); unsupported and misplaced: no shown line does the "already in markers -> refuse" check, and iterencode only hands markers to c_make_encoder/make_iterencode (encoder.py:251-260), so the detection itself lives in make_iterencode, which the model was not shown

**Your call:** _(agree / change to …)_

---

## q16 (location, chatter) — flagged

**Question:** How does the tool cope with a source file that has a syntax error halfway through?

**Classification:** answered

**Answer:**

> The tool handles a source file with a syntax error halfway through by extracting valid code chunks up to and including the error, ensuring that the syntax error does not cause the entire file to be discarded. Specifically, it preserves the module and valid code segments before the error, and it correctly identifies and isolates syntactically valid components such as functions, methods, and variables. For instance, in a test case, a broken method definition (e.g., `def broken(:`) causes the parser to stop at that point, and the rest of the file is not included, but the valid code before and after the error is preserved as separate chunks. The tool ensures that the module chunk and its valid components survive, and syntax errors do not cause the loss of subsequent valid code. This behavior is validated in the test `test_module_chunk_survives_syntax_errors`, which confirms that the module's docstring and valid assignments (like `BEFORE = 1` and `AFTER = 2`) are preserved, while invalid or broken syntax (like `def broken(:`) is excluded [C1]. Additionally, the `assert_source_consistent` function verifies that the extracted chunks maintain the correct line order and content, ensuring integrity of the source [C2].

**Relevant (eval set):** `src/chatter/extract.py::_extract_segmented`, `src/chatter/extract.py::_split_segments`, `src/chatter/extract.py::_recover_broken`

**Shown context:**
- [C1] `tests/test_extract.py::test_module_chunk_survives_syntax_errors` tests/test_extract.py:685-699 (cited)
- [C2] `tests/test_extract.py::assert_source_consistent` tests/test_extract.py:27-42 (cited)
- [C3] `src/chatter/cli.py::<module>` src/chatter/cli.py:1-5,7-7,9-13,15-15,17-37,39-40,244-244
- [C4] `src/chatter/cli.py::make_app` src/chatter/cli.py:47-189
- [C5] `src/chatter/cli.py::make_app.ask` src/chatter/cli.py:122-187
- [C6] `src/chatter/index.py::build_index` src/chatter/index.py:332-408
- [C7] `src/chatter/cli.py::make_app.index` src/chatter/cli.py:60-99
- [C8] `tests/test_extract.py::test_broken_method_does_not_swallow_rest_of_file` tests/test_extract.py:309-331

[C1] tests/test_extract.py (shown lines near the grader’s citations):
```python
  685  def test_module_chunk_survives_syntax_errors() -> None:
  686      m = module_chunk('''
  687          """Doc."""
  688          import os
  689          BEFORE = 1
  690  
  691          def broken(:
  692              pass
  693  
```

**Verdict:** partial
**Justification:** supported: the module chunk keeps the docstring, BEFORE = 1 and AFTER = 2 and drops broken/BAD (tests/test_extract.py:685-699), and assert_source_consistent checks line order and content (test_extract.py:27-42); contradicted: "extracting chunks up to and including the error" and "the parser stops at that point, and the rest of the file is not included", because the shown broken.py test keeps Broken.also_ok, good_after and good_async after the broken method (test_extract.py:309-331); the extract.py recovery code was not shown

**Your call:** _(agree / change to …)_

---

## h03 (behavior, stdlib) — flagged

**Question:** How does the built-in HTTP server decide whether to keep a connection open for another request or close it?

**Classification:** answered

**Answer:**

> The built-in HTTP server decides whether to keep a connection open or close it based on the HTTP/1.1 "Connection" header. Specifically, if the server receives a "Connection: close" header, the connection will be closed after the response. Otherwise, the connection is assumed to remain open. For HTTP/1.0, the presence of a "Keep-Alive" header indicates a persistent connection, while the absence of such a header implies the connection will close. Additionally, if no explicit "Connection" header is present, the server defaults to closing the connection after the response. This behavior is implemented in the `_check_close` method of the `HTTPResponse` class, which checks for these headers and determines whether the connection will close at the end of the response. [C2, C4, C7]

**Relevant (eval set):** `http/server.py::BaseHTTPRequestHandler.parse_request`, `http/server.py::BaseHTTPRequestHandler.handle`

**Shown context:**
- [C1] `http/client.py::<module>` http/client.py:1-69,71-80,82-90,92-93,95-95,97-100,106-108,110-112,137-140,142-147,152-154,156-158,1549-1550
- [C2] `http/client.py::HTTPResponse._check_close` http/client.py:396-423 (cited)
- [C3] `http/client.py::HTTPResponse.isclosed` http/client.py:453-461
- [C4] `http/client.py::HTTPResponse` http/client.py:252-803 (cited)
- [C5] `http/client.py::HTTPResponse.close` http/client.py:430-435
- [C6] `http/client.py::HTTPConnection.getresponse` http/client.py:1386-1447
- [C7] `http/client.py::HTTPResponse.begin` http/client.py:324-394 (cited)
- [C8] `http/client.py::HTTPResponse.__init__` http/client.py:261-289

[C2] http/client.py (shown lines near the grader’s citations):
```python
  396      def _check_close(self):
  397          conn = self.headers.get("connection")
  398          if self.version == 11:
  399              # An HTTP/1.1 proxy is assumed to stay open unless
  400              # explicitly closed.
  401              if conn and "close" in conn.lower():
  402                  return True
  403              return False
  404  
```

[C4] http/client.py (shown lines near the grader’s citations):
```python
  394              self.will_close = True
  395  
  396      def _check_close(self):
  397          conn = self.headers.get("connection")
  398          if self.version == 11:
  399              # An HTTP/1.1 proxy is assumed to stay open unless
  400              # explicitly closed.
  401              if conn and "close" in conn.lower():
  402                  return True
  403              return False
  404  
```

**Verdict:** wrong
**Justification:** it presents http.client's HTTPResponse.check_close (a client reading a server's response, http/client.py:396-423) as the built-in server's logic, though no http/server.py code was shown; it also contradicts itself ("otherwise assumed open" vs "defaults to closing"), whereas check_close keeps HTTP/1.1 open unless Connection: close (client.py:398-403)

**Your call:** _(agree / change to …)_

---

## h04 (location, stdlib) — flagged

**Question:** What stops a malicious server from sending an endless list of response headers to the HTTP client?

**Classification:** answered

**Answer:**

> The HTTP client enforces a state machine that prevents a malicious server from sending an endless list of response headers. Specifically, the client only allows a new request to be sent after the response headers have been fully read and processed. This is enforced by the `HTTPConnection` class, which ensures that the client cannot issue a new request until it has read the response headers. As stated in the code, "a second request may not be started until {response-headers-read}" [C1]. This mechanism prevents the client from being overwhelmed by an endless stream of headers, as it ensures that the client only proceeds to the next request after confirming the response headers have been received and parsed. Additionally, the `HTTPResponse.begin()` method reads headers until a non-100 response is encountered, which stops the client from processing an infinite loop of 100 Continue responses [C5].

**Relevant (eval set):** `http/client.py::_read_headers`, `http/client.py::parse_headers`

**Shown context:**
- [C1] `http/client.py::<module>` http/client.py:1-69,71-80,82-90,92-93,95-95,97-100,106-108,110-112,137-140,142-147,152-154,156-158,1549-1550 (cited)
- [C2] `http/server.py::BaseHTTPRequestHandler` http/server.py:350-394
- [C3] `http/client.py::HTTPConnection` http/client.py:1379-1433
- [C4] `http/client.py::HTTPResponse` http/client.py:356-413
- [C5] `http/client.py::HTTPResponse.begin` http/client.py:324-394 (cited)
- [C6] `http/server.py::BaseHTTPRequestHandler.parse_request` http/server.py:267-375
- [C7] `http/client.py::HTTPResponse._check_close` http/client.py:396-423
- [C8] `http/client.py::HTTPConnection._send_request` http/client.py:1340-1384

[C1] http/client.py (shown lines near the grader’s citations):
```python
   43  
   44  This diagram presents the following rules:
   45    -- a second request may not be started until {response-headers-read}
   46    -- a response [object] cannot be retrieved until {request-sent}
   47    -- there is no differentiation between an unread response body and a
```

[C2] http/server.py (shown lines near the grader’s citations):
```python
  350                  HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
  351                  "Line too long",
  352                  str(err))
  353              return False
```

[C6] http/server.py (shown lines near the grader’s citations):
```python
  343  
  344          # Examine the headers and look for a Connection directive.
  345          try:
  346              self.headers = http.client.parse_headers(self.rfile,
  347                                                       _class=self.MessageClass)
  348          except http.client.LineTooLong as err:
  349              self.send_error(
  350                  HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
  351                  "Line too long",
  352                  str(err))
  353              return False
```

**Verdict:** wrong
**Justification:** the request state machine ("a second request may not be started until {response-headers-read}", http/client.py:45) has nothing to do with header count, and begin() keeps looping while 100 Continue responses arrive instead of stopping them (client.py:330-338); it ignored the shown MAXHEADERS = 100 (client.py:112) and the server's "Too many headers" HTTPException handling around parse_headers (http/server.py:345-359)

**Your call:** _(agree / change to …)_

---

## h15 (behavior, chatter) — flagged

**Question:** A function is too long to fit in the prompt. Which lines of it does the model actually get to see?

**Classification:** answered

**Answer:**

> When a function is too long to fit in the prompt, the model truncates oversized lines and processes only the portions that fit within the token budget. Specifically, if a single line exceeds the budget, it becomes its own part and is truncated by the model, as seen in the test case where a line longer than the budget is cut and marked with " …" [C4]. Additionally, the model only processes lines that fit within the token limit, and for a large function, only the relevant lines (based on lexical or semantic matching) are shown, such as in the test where only lines related to "summarize results" are shown, and the rest are excluded [C6]. The exact lines seen depend on the budget and the content, with the model ensuring that no single line exceeds the token limit, and only the first and last lines of a chunk that fit within the budget are processed [C2, C7].

**Relevant (eval set):** `src/chatter/answer.py::truncation_window`, `src/chatter/answer.py::_truncated_block`, `src/chatter/answer.py::_region`, `src/chatter/answer.py::_line_score`

**Shown context:**
- [C1] `tests/test_answer.py::test_question_too_long_for_budget` tests/test_answer.py:201-203
- [C2] `tests/test_index.py::test_single_line_longer_than_budget_becomes_its_own_part` tests/test_index.py:151-154 (cited)
- [C3] `tests/test_cli.py::test_ask_context_too_small` tests/test_cli.py:282-285
- [C4] `tests/test_answer.py::test_line_longer_than_allowance_is_cut_by_characters` tests/test_answer.py:144-147 (cited)
- [C5] `tests/test_answer.py::big_function` tests/test_answer.py:54-59
- [C6] `tests/test_answer.py::test_dense_only_hit_with_lexical_match_is_anchored_too` tests/test_answer.py:108-111 (cited)
- [C7] `src/chatter/index.py::plan_parts` src/chatter/index.py:151-167 (cited)
- [C8] `src/chatter/embed.py::<module>` src/chatter/embed.py:1-5,7-7,9-12,14-15

[C2] tests/test_index.py (shown lines near the grader’s citations):
```python
  151  def test_single_line_longer_than_budget_becomes_its_own_part() -> None:
  152      (c,) = chunk("def f():\n    x = 1\n    y = [" + ", ".join(["a"] * 200) + "]\n    z = 2\n")
  153      (parts,) = plan_parts([c], HashEmbedder(max_tokens=30))
  154      assert [(p.first, p.last) for p in parts] == [(0, 1), (2, 2), (3, 3)]
```

[C4] tests/test_answer.py (shown lines near the grader’s citations):
```python
  144  def test_line_longer_than_allowance_is_cut_by_characters() -> None:
  145      chunk = make_chunk("def f():\n    data = [" + ", ".join(f"v{i}" for i in range(2000)) + "]\n")
  146      block = only_block("data", make_hit(chunk), 200)
  147      assert block.truncated and block.text.count("\n") <= 4 and " …" in block.text
```

[C7] src/chatter/index.py (shown lines near the grader’s citations):
```python
  151  def plan_parts(chunks: Sequence[Chunk], embedder: Embedder) -> list[list[EmbeddingPart]]:
  152      """Split each chunk into parts that fit ``embedder.max_tokens``.
  153  
  154      Whole texts are counted in one batch; per-line counts are only computed
  155      for oversized chunks. A single line longer than the budget becomes its
  156      own part and is truncated by the model.
  157      """
  158      texts = [embedding_text(chunk) for chunk in chunks]
  159      totals = embedder.count_tokens(texts)
```

**Verdict:** partial
**Justification:** supported: a single line longer than the allowance is cut by characters and marked " …" (tests/test_answer.py:144-147), and a lexically matching line is shown while the head is not (test_answer.py:108-111); wrong or unsupported: it mixes embedding-part splitting (plan_parts, "becomes its own part and is truncated by the model", src/chatter/index.py:151-167; tests/test_index.py:151-154) into prompt truncation, and "only the first and last lines ... are processed" misreads the (first, last) part indices; the truncation_window logic was not shown

**Your call:** _(agree / change to …)_

---

## q17 (location, chatter) — flagged

**Question:** Why are line and column positions read by index rather than by attribute name?

**Classification:** abstained

**Answer:**

> The retrieved code does not contain the answer.

**Relevant (eval set):** `src/chatter/extract.py::_start_row`, `src/chatter/extract.py::_end_row`

**Shown context:**
- [C1] `src/chatter/index.py::IndexStats` src/chatter/index.py:70-76
- [C2] `src/chatter/cli.py::make_app.index` src/chatter/cli.py:60-99
- [C3] `src/chatter/answer.py::ContextBlock.location` src/chatter/answer.py:121-124
- [C4] `src/chatter/extract.py::Chunk` src/chatter/extract.py:53-83
- [C5] `src/chatter/index.py::build_index` src/chatter/index.py:332-408
- [C6] `src/chatter/index.py::IndexConfig` src/chatter/index.py:62-67
- [C7] `src/chatter/extract.py::_end_row` src/chatter/extract.py:218-223 (relevant)
- [C8] `src/chatter/answer.py::format_line_ranges` src/chatter/answer.py:329-341

[C7] src/chatter/extract.py (shown lines near the grader’s citations):
```python
  218  def _end_row(node: Node) -> int:
  219      """Last 0-based row actually occupied by ``node``."""
  220      end_row, end_column = node.end_point[0], node.end_point[1]
  221      if end_column == 0 and end_row > _start_row(node):
  222          return end_row - 1
  223      return end_row
```

**Verdict:** correctly abstained
**Justification:** the shown end_row reads node.end_point[0]/[1] by index (src/chatter/extract.py:218-223) but gives no reason; the explanatory comment at extract.py:209-211 is not in any shown chunk

**Your call:** _(agree / change to …)_

---

## h02 (behavior, stdlib) — flagged

**Question:** json duplicate keys which one wins

**Classification:** abstained

**Answer:**

> The retrieved code does not contain the answer.

**Relevant (eval set):** `json/decoder.py::JSONObject`

**Shown context:**
- [C1] `json/encoder.py::JSONEncoder.__init__` json/encoder.py:105-159
- [C2] `json/__init__.py::dumps` json/__init__.py:183-238
- [C3] `json/encoder.py::JSONEncoder` json/encoder.py:74-261
- [C4] `json/__init__.py::<module>` json/__init__.py:1-102,104-104,106-108,110-118,241-241
- [C5] `json/decoder.py::JSONObject` json/decoder.py:137-219 (relevant)
- [C6] `json/decoder.py::JSONDecoder` json/decoder.py:262-364
- [C7] `json/encoder.py::JSONEncoder.encode` json/encoder.py:183-203
- [C8] `json/decoder.py::py_scanstring` json/decoder.py:70-127

[C5] json/decoder.py (shown lines near the grader’s citations):
```python
  188          except StopIteration as err:
  189              raise JSONDecodeError("Expecting value", s, err.value) from None
  190          pairs_append((key, value))
  191          try:
  192              nextchar = s[end]
   …
  211              raise JSONDecodeError(
  212                  "Expecting property name enclosed in double quotes", s, end - 1)
  213      if object_pairs_hook is not None:
  214          result = object_pairs_hook(pairs)
  215          return result, end
  216      pairs = dict(pairs)
  217      if object_hook is not None:
  218          pairs = object_hook(pairs)
  219      return pairs, end
```

**Verdict:** wrong
**Justification:** false abstention: the shown JSONObject appends each (key, value) pair in order and then does pairs = dict(pairs), so the last duplicate wins unless object_pairs_hook is given the full pair list (json/decoder.py:190, 213-219)

**Your call:** _(agree / change to …)_

---

## q03 (behavior, stdlib) — flagged

**Question:** When an asyncio program's main coroutine finishes, what happens to background tasks that are still running?

**Classification:** mixed

**Answer:**

> The provided code does not describe the behavior of background tasks when a main coroutine finishes. Therefore, the retrieved code does not contain the answer.

**Relevant (eval set):** `asyncio/runners.py::_cancel_all_tasks`, `asyncio/runners.py::Runner.close`

**Shown context:**
- [C1] `asyncio/runners.py::run` asyncio/runners.py:160-195
- [C2] `asyncio/runners.py::Runner` asyncio/runners.py:20-157
- [C3] `asyncio/staggered.py::staggered_race` asyncio/staggered.py:13-174
- [C4] `asyncio/runners.py::Runner.run` asyncio/runners.py:86-129
- [C5] `asyncio/taskgroups.py::TaskGroup` asyncio/taskgroups.py:12-278
- [C6] `asyncio/tasks.py::_AsCompletedIterator` asyncio/tasks.py:570-634
- [C7] `asyncio/staggered.py::staggered_race.run_one_coro` asyncio/staggered.py:91-145
- [C8] `asyncio/tasks.py::as_completed` asyncio/tasks.py:637-688

[C2] asyncio/runners.py (shown lines near the grader’s citations):
```python
   62          self.close()
   63  
   64      def close(self):
   65          """Shutdown and close event loop."""
   66          if self._state is not _State.INITIALIZED:
   67              return
   68          try:
   69              loop = self._loop
   70              _cancel_all_tasks(loop)
   71              loop.run_until_complete(loop.shutdown_asyncgens())
   72              loop.run_until_complete(
```

**Verdict:** wrong
**Justification:** false abstention: shown Runner.close calls cancel_all_tasks(loop) before closing the loop (asyncio/runners.py:64-79, call at :70), _exit_ calls close (runners.py:61-62), and run() executes main inside `with Runner(...)` (runners.py:194-195), so leftover tasks are cancelled when main returns

**Your call:** _(agree / change to …)_

---

## q19 (behavior, chatter) — flagged

**Question:** Why doesn't a word that appears in almost every function get a negative keyword score?

**Classification:** mixed

**Answer:**

> The provided code does not contain information explaining why a word that appears in almost every function does not get a negative keyword score. The relevant code discusses BM25 scoring and IDf calculations, including the use of Lucene's IDf formula, which ensures positive scores for matches and zero for non-matches [C3]. However, there is no explicit explanation or mechanism in the code that addresses the behavior of a word appearing in almost every function resulting in a non-negative score. Therefore, based on the provided chunks, the reasoning behind this behavior cannot be determined.
>
> The retrieved code does not contain the answer.

**Relevant (eval set):** `src/chatter/retrieve.py::PositiveIdfBM25`, `src/chatter/retrieve.py::PositiveIdfBM25._calc_idf`

**Shown context:**
- [C1] `src/chatter/answer.py::_line_score` src/chatter/answer.py:263-264
- [C2] `tests/test_retrieve.py::test_positive_idf_bm25_scores_matches_above_zero_and_others_zero` tests/test_retrieve.py:41-47
- [C3] `src/chatter/retrieve.py::PositiveIdfBM25` src/chatter/retrieve.py:53-63 (cited, relevant)
- [C4] `src/chatter/answer.py::query_terms` src/chatter/answer.py:154-155
- [C5] `src/chatter/retrieve.py::BM25Index` src/chatter/retrieve.py:66-79
- [C6] `src/chatter/answer.py::truncation_window` src/chatter/answer.py:217-249
- [C7] `tests/test_retrieve.py::test_okapi_scores_real_matches_at_or_below_zero_on_tiny_corpora` tests/test_retrieve.py:31-38
- [C8] `tests/test_answer.py::test_best_line_wins_by_distinct_query_terms_earliest_on_tie` tests/test_answer.py:88-97

[C3] src/chatter/retrieve.py (shown lines near the grader’s citations):
```python
   53  class PositiveIdfBM25(BM25Okapi):
   54      """BM25Okapi with Lucene's idf, ``log(1 + (N - n + 0.5) / (n + 0.5))``.
   55  
   56      Okapi's idf is <= 0 for terms in at least half the documents (always the
   57      case in a one-file repo), so real matches could score zero or negative.
   58      This idf is strictly positive: matches score > 0, non-matches exactly 0.
   59      """
   60  
   61      def _calc_idf(self, nd: Mapping[str, int]) -> None:
```

**Verdict:** partial
**Justification:** it names the mechanism correctly (PositiveIdfBM25 uses Lucene's idf, so matches score > 0 and non-matches 0, src/chatter/retrieve.py:53-63) but then wrongly says the code gives no explanation and abstains, although the docstring explains that Okapi's idf is <= 0 for terms in at least half the documents (retrieve.py:56-58)

**Your call:** _(agree / change to …)_

---

## h07 (location, stdlib) — flagged

**Question:** When an email with attachments is written out, how is the separator between the parts chosen so it can't accidentally appear inside the content?

**Classification:** mixed

**Answer:**

> The provided code does not describe how the separator between parts in an email is chosen or how it is ensured not to appear inside the content. Therefore, the retrieved code does not contain the answer.

**Relevant (eval set):** `email/generator.py::Generator._make_boundary`, `email/generator.py::Generator._handle_multipart`

**Shown context:**
- [C1] `email/feedparser.py::FeedParser` email/feedparser.py:136-530
- [C2] `email/message.py::MIMEPart.iter_attachments` email/message.py:1066-1119
- [C3] `email/message.py::MIMEPart` email/message.py:981-1201
- [C4] `email/message.py::MIMEPart.iter_parts` email/message.py:1121-1127
- [C5] `email/message.py::MIMEPart.is_attachment` email/message.py:1009-1011
- [C6] `email/feedparser.py::FeedParser._parsegen` email/feedparser.py:218-470
- [C7] `email/message.py::MIMEPart.add_attachment` email/message.py:1191-1192
- [C8] `email/message.py::Message.is_multipart` email/message.py:211-213

[C1] email/feedparser.py (shown lines near the grader’s citations):
```python
  304              return
  305          if self._cur.get_content_maintype() == 'multipart':
  306              boundary = self._cur.get_boundary()
  307              if boundary is None:
  308                  # The message /claims/ to be a multipart but it has not
  309                  # defined a boundary.  That's a problem which we'll handle by
  310                  # reading everything until the EOF and marking the message as
  311                  # defective.
  312                  defect = errors.NoBoundaryInMultipartDefect()
  313                  self.policy.handle_defect(self._cur, defect)
  314                  lines = []
```

[C3] email/message.py (shown lines near the grader’s citations):
```python
 1137          content_manager.set_content(self, *args, **kw)
 1138  
 1139      def _make_multipart(self, subtype, disallowed_subtypes, boundary):
 1140          if self.get_content_maintype() == 'multipart':
 1141              existing_subtype = self.get_content_subtype()
 1142              disallowed_subtypes = disallowed_subtypes + (subtype,)
 1143              if existing_subtype in disallowed_subtypes:
 1144                  raise ValueError("Cannot convert {} to {}".format(
 1145                      existing_subtype, subtype))
 1146          keep_headers = []
 1147          part_headers = []
```

**Verdict:** correctly abstained
**Justification:** the shown chunks cover parsing boundaries (email/feedparser.py:306-440) and MIMEPart helpers that only set a boundary the caller passes in (email/message.py:1139-1173); how the generator chooses a boundary (make_boundary) is not shown

**Your call:** _(agree / change to …)_

---

## h17 (flow, chatter) — flagged

**Question:** Walk me through everything that happens between typing chatter ask and seeing the answer printed with its sources.

**Classification:** mixed

**Answer:**

> 1. The user types `chatter ask` with a question, optional `k` (number of chunks to retrieve), `path` (repository root), `model` (Hugging Face answer model), `max_context_tokens`, and `max_new_tokens`.
>
> 2. The CLI invokes the `ask` command, which first retrieves relevant code chunks using the retriever. It calls `_open_retriever(path, embedder_factory).search(question, k=k)` to find the most relevant chunks based on the query and top-k limit [C4, C5].
>
> 3. If no hits are found, the CLI outputs "No indexed code matched the question; nothing to answer from" and exits [C4, C5].
>
> 4. Otherwise, it loads an answer model using `_load(...)` with the provided `model` name, which is passed to `generator_factory(GeneratorConfig(model_name=model))` [C4, C5].
>
> 5. The `answer_question` function is called with the question, hits, and generator. It first builds the prompt context by calling `build_context(question, hits, generator, max_context_tokens=max_context_tokens)` [C3, C7]. This function allocates a budget of tokens to each hit based on the `max_context_tokens` limit, ensuring no single chunk exceeds its fair share. If a chunk's share falls below `MIN_BLOCK_TOKENS`, it is dropped and not included in the final prompt [C2, C7].
>
> 6. The prompt is constructed with the user's question and the relevant code chunks, each wrapped in `<chunk tag="C1" ...>` format [C2, C3]. The prompt includes metadata such as chunk tags, file:line locations, and chunk IDs.
>
> 7. The generator produces a streamed response, which is passed to `on_text` (a lambda that prints each piece without a newline) to display the answer incrementally [C4, C3].
>
> 8. During generation, a `RepetitionGuard` monitors for loops. If a substantial line appears three times or a repeated phrase (8–80 characters) appears six times, generation stops and the model's output is truncated [C6].
>
> 9. After generation, the result is processed:
>    - If the model stopped due to repetition, a warning is printed: "[generation stopped: the model started repeating itself]" [C4].
>    - If any chunks were dropped due to context budget, a warning is printed indicating how many were dropped [C4].
>    - If the model returned `AnswerStatus.ABSTAINED`, the CLI prints "NOT FOUND: the retrieved code does not contain the answer" and lists the searched chunks with their tags and locations [C4, C5].
>    - If the model returned `AnswerStatus.MIXED`, a note is printed indicating the retrieved code does not fully answer the question [C4].
>    - Otherwise, the answer is printed with citations. Each citation includes the chunk's tag, chunk ID, and location. If a chunk was truncated, a "(truncated)" note is appended [C4, C5].
>
> 10. If the answer cites unknown tags, a warning is printed listing them [C4].
>
> 11. The final output includes the answer and, if applicable, citations to the source code chunks, with tags cited inline (e.g., [C1]) and mapped to file:line locations [C2, C4].
>
> This entire process ensures that the answer is generated from relevant, contextually appropriate code chunks, with proper handling of edge cases like no matches, repetition, or abstinence.

**Relevant (eval set):** `src/chatter/cli.py::make_app.ask`, `src/chatter/retrieve.py::Retriever.search`, `src/chatter/answer.py::answer_question`, `src/chatter/answer.py::build_context`, `src/chatter/answer.py::build_user_message`, `src/chatter/answer.py::HFGenerator.generate`, `src/chatter/answer.py::RepetitionGuard.feed`, `src/chatter/answer.py::classify_answer`, `src/chatter/answer.py::parse_citations`

**Shown context:**
- [C1] `src/chatter/cli.py::make_app.search` src/chatter/cli.py:101-120 (cited)
- [C2] `src/chatter/answer.py::<module>` src/chatter/answer.py:1-45,47-47,49-54,56-58,60-64,66-76,78-83 (cited)
- [C3] `src/chatter/answer.py::answer_question` src/chatter/answer.py:457-496 (cited, relevant)
- [C4] `src/chatter/cli.py::make_app.ask` src/chatter/cli.py:122-187 (cited, relevant)
- [C5] `src/chatter/cli.py::make_app` src/chatter/cli.py:47-189 (cited)
- [C6] `src/chatter/answer.py::RepetitionGuard` src/chatter/answer.py:386-449 (cited)
- [C7] `src/chatter/answer.py::build_context` src/chatter/answer.py:163-195 (cited, relevant)
- [C8] `src/chatter/answer.py::AnswerResult` src/chatter/answer.py:133-142

[C3] src/chatter/answer.py (shown lines near the grader’s citations):
```python
  457  def answer_question(
  458      question: str,
  459      hits: Sequence[Hit],
  460      generator: Generator,
  461      *,
  462      max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
  463      max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
  464      on_text: Callable[[str], None] = lambda _: None,
  465  ) -> AnswerResult:
```

[C4] src/chatter/cli.py (shown lines near the grader’s citations):
```python
  122      @app.command()
  123      def ask(
  124          question: Annotated[str, typer.Argument(help="Question about the code.")],
  125          k: Annotated[int, typer.Option("-k", "--top-k", min=1, help="Chunks to retrieve.")] = 8,
  126          path: Annotated[Path, typer.Option(help="Repository root containing .chatter.")] = Path("."),
  127          model: Annotated[str, typer.Option(help="Hugging Face answer model.")] = DEFAULT_ANSWER_MODEL,
  128          max_context_tokens: Annotated[
  129              int, typer.Option(min=256, help="Token budget for the whole prompt.")
  130          ] = DEFAULT_MAX_CONTEXT_TOKENS,
```

[C5] src/chatter/cli.py (shown lines near the grader’s citations):
```python
  120              )
  121  
  122      @app.command()
  123      def ask(
  124          question: Annotated[str, typer.Argument(help="Question about the code.")],
  125          k: Annotated[int, typer.Option("-k", "--top-k", min=1, help="Chunks to retrieve.")] = 8,
  126          path: Annotated[Path, typer.Option(help="Repository root containing .chatter.")] = Path("."),
  127          model: Annotated[str, typer.Option(help="Hugging Face answer model.")] = DEFAULT_ANSWER_MODEL,
  128          max_context_tokens: Annotated[
  129              int, typer.Option(min=256, help="Token budget for the whole prompt.")
  130          ] = DEFAULT_MAX_CONTEXT_TOKENS,
```

**Verdict:** correct
**Justification:** the steps match the shown code: ask options, open_retriever().search, the no-hits message, load of the generator, answer_question -> build_context, streaming via on_text, RepetitionGuard (3 lines / 6 phrases), then the repetition, dropped, NOT FOUND, MIXED, citation and unknown-tag output (src/chatter/cli.py:122-187; src/chatter/answer.py:163-195, 386-449, 457-496); small slips: below MIN_BLOCK_TOKENS the lowest-ranked chunk is dropped, not necessarily the one whose share is small (answer.py:184-185, 15-16), and the file:line/chunk-id prompt metadata is not in shown lines

**Your call:** _(agree / change to …)_

---

## q07 (behavior, stdlib) — flagged

**Question:** A subject line looks like =?utf-8?b?SGVsbG8=?= . How do I turn that into readable text?

**Classification:** near_miss_abstention

**Answer:**

> The provided code does not contain the answer.

**Relevant (eval set):** `email/header.py::decode_header`

**Shown context:**
- [C1] `email/message.py::_formatparam` email/message.py:39-71
- [C2] `email/quoprimime.py::decode` email/quoprimime.py:232-276
- [C3] `email/header.py::_ValueFormatter.feed` email/header.py:438-473
- [C4] `email/header.py::_ValueFormatter` email/header.py:408-533
- [C5] `email/quoprimime.py::body_encode` email/quoprimime.py:153-226
- [C6] `email/charset.py::Charset.body_encode` email/charset.py:369-398
- [C7] `email/header.py::Header.encode` email/header.py:309-387
- [C8] `email/contentmanager.py::_encode_text` email/contentmanager.py:144-180

[C2] email/quoprimime.py (shown lines near the grader’s citations):
```python
  232  def decode(encoded, eol=NL):
  233      """Decode a quoted-printable string.
  234  
  235      Lines are separated with eol, which defaults to \\n.
  236      """
  237      if not encoded:
  238          return encoded
  239      # BAW: see comment in encode() above.  Again, we're building up the
  240      # decoded string with string concatenation, which could be done much more
```

**Verdict:** correctly abstained
**Justification:** the shown chunks only encode (Header.encode, ValueFormatter, body_encode, formatparam) or decode quoted-printable bodies (email/quoprimime.py:232-276); nothing shown decodes an RFC 2047 encoded word like =?utf-8?b?...?= (decode_header was not retrieved)

**Your call:** _(agree / change to …)_

---

## h01 (behavior, stdlib) — random (seed 20261115)

**Question:** json.loads keeps giving me 'Extra data' on a file that looks fine to me. What does that error actually mean and where is it raised?

**Classification:** answered

**Answer:**

> The error "Extra data" in `json.loads` means that the input string contains data after the valid JSON document, which is not part of the JSON structure. This typically happens when the input has trailing characters or is malformed, such as having extra content after the closing bracket or brace of a JSON object or array.
>
> This error is raised in the `decode` method of the `JSONDecoder` class, specifically in the `decode` method where it checks if the end of the parsed string matches the length of the input string. If there's extra data, it raises a `JSONDecodeError` with the message "Extra data" [C6].
>
> The error is further detailed in the `JSONDecodeError` class, which includes properties like `doc`, `pos`, `lineno`, and `colno` to help pinpoint the exact location of the error in the input [C2, C4]. The `JSONDecodeError` is raised when the parser detects that the input string has content beyond the valid JSON document, which is not allowed per JSON specification.

**Relevant (eval set):** `json/decoder.py::JSONDecoder.decode`

**Shown context:**
- [C1] `json/decoder.py::JSONObject` json/decoder.py:137-219
- [C2] `json/decoder.py::JSONDecodeError` json/decoder.py:20-43 (cited)
- [C3] `json/tool.py::main` json/tool.py:18-82
- [C4] `json/decoder.py::JSONDecodeError.__init__` json/decoder.py:31-40 (cited)
- [C5] `json/decoder.py::JSONArray` json/decoder.py:221-259
- [C6] `json/decoder.py::JSONDecoder.decode` json/decoder.py:340-349 (cited, relevant)
- [C7] `json/decoder.py::py_scanstring` json/decoder.py:70-127
- [C8] `json/__init__.py::loads` json/__init__.py:299-359

[C6] json/decoder.py (shown lines near the grader’s citations):
```python
  340      def decode(self, s, _w=WHITESPACE.match):
  341          """Return the Python representation of ``s`` (a ``str`` instance
  342          containing a JSON document).
  343  
  344          """
  345          obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  346          end = _w(s, end).end()
  347          if end != len(s):
  348              raise JSONDecodeError("Extra data", s, end)
```

**Verdict:** correct
**Justification:** JSONDecoder.decode raw_decodes, skips trailing whitespace, and raises JSONDecodeError("Extra data", s, end) when end != len(s) (json/decoder.py:340-349); the position attributes come from JSONDecodeError (decoder.py:20-40)

**Your call:** _(agree / change to …)_

---

## h14 (location, chatter) — random (seed 20261115)

**Question:** Where do the [C1]-style labels in an answer get turned back into real file names and line numbers?

**Classification:** answered

**Answer:**

> The [C1]-style labels in an answer are turned back into real file names and line numbers in the `ContextBlock.location` property, which formats the output as `path:a-b` or `path:a-b,c-d` based on the shown line ranges [C2, C6]. The line ranges are derived from the `shown_lines` tuple, which is populated during the truncation or full block rendering process [C8, C5]. Specifically, the `format_line_ranges` function converts the sequence of line numbers into a comma-separated range format (e.g., '1-3,7-8') [C7]. This formatting is then used in the `location` property to produce the final path and line number representation.

**Relevant (eval set):** `src/chatter/answer.py::parse_citations`, `src/chatter/answer.py::ContextBlock.location`, `src/chatter/cli.py::make_app.ask`

**Shown context:**
- [C1] `src/chatter/answer.py::<module>` src/chatter/answer.py:1-45,47-47,49-54,56-58,60-64,66-76,78-83 (cited)
- [C2] `src/chatter/answer.py::ContextBlock` src/chatter/answer.py:109-124 (cited)
- [C3] `tests/test_cli.py::test_ask_prints_answer_and_citations` tests/test_cli.py:203-212
- [C4] `src/chatter/answer.py::truncation_window` src/chatter/answer.py:217-249
- [C5] `src/chatter/answer.py::_full_block` src/chatter/answer.py:267-275 (cited)
- [C6] `src/chatter/answer.py::ContextBlock.location` src/chatter/answer.py:121-124 (cited, relevant)
- [C7] `src/chatter/answer.py::format_line_ranges` src/chatter/answer.py:329-341 (cited)
- [C8] `src/chatter/answer.py::_truncated_block` src/chatter/answer.py:278-300 (cited)

[C2] src/chatter/answer.py (shown lines near the grader’s citations):
```python
  109  @dataclass(frozen=True, slots=True)
  110  class ContextBlock:
  111      tag: str  # "C1"
  112      hit: Hit
  113      shown_lines: tuple[int, ...]  # file line numbers shown to the model
  114      truncated: bool
  115      text: str  # the rendered <chunk> block
  116  
  117      @property
```

**Verdict:** correct
**Justification:** each ContextBlock carries its tag plus a location property that formats path:ranges from shown_lines (src/chatter/answer.py:109-124) via format_line_ranges (answer.py:329-341), with shown_lines set in full_block/truncated_block (answer.py:267-300); it leaves out the citation-parsing/CLI step named in the module docstring (answer.py:4-6)

**Your call:** _(agree / change to …)_

---

## q05 (behavior, stdlib) — random (seed 20261115)

**Question:** How does the HTTP client read a response body when the server doesn't say how long it is and sends it in pieces?

**Classification:** answered

**Answer:**

> When the server does not specify the length of the response body and sends it in pieces (using chunked transfer encoding), the HTTP client reads the response body using chunked transfer encoding. Specifically, the `HTTPResponse` class handles this through its `_read_chunked` method, which reads each chunk from the response body in a loop until the entire body is consumed.
>
> The process works as follows:
> - The client checks if the response uses chunked transfer encoding by examining the `transfer-encoding` header. If it is set to "chunked", the `chunked` flag in the `HTTPResponse` object is set to `True` [C5].
> - The `_read_chunked` method is then used to read the response body in chunks. It reads each chunk size from the response, which is specified in the chunked encoding format, and appends it to the result [C7].
> - The method reads the chunk size, then reads the actual data of that size from the underlying file pointer (`self.fp`), and continues until all chunks are processed.
> - If the client requests a limited number of bytes (`amt`), it stops reading once the requested amount is reached, or when the end of the response is reached [C7].
>
> This approach allows the client to read the response body in pieces without needing to know the total size in advance, which is essential for handling streaming or large responses. The client also manages the connection state appropriately, such as closing the connection when the response ends or when the server indicates it should close the connection [C5, C4, C6].
>
> Thus, the HTTP client reads the response body in pieces using chunked transfer encoding, reading each chunk as it arrives and managing the connection accordingly.

**Relevant (eval set):** `http/client.py::HTTPResponse._read_chunked`, `http/client.py::HTTPResponse._read_next_chunk_size`, `http/client.py::HTTPResponse._get_chunk_left`

**Shown context:**
- [C1] `http/client.py::HTTPResponse.read` http/client.py:463-501
- [C2] `http/server.py::BaseHTTPRequestHandler` http/server.py:189-237
- [C3] `http/client.py::HTTPConnection` http/client.py:1379-1433
- [C4] `http/client.py::HTTPResponse` http/client.py:480-541 (cited)
- [C5] `http/client.py::HTTPResponse.begin` http/client.py:324-394 (cited)
- [C6] `http/client.py::HTTPResponse.readinto` http/client.py:503-535 (cited)
- [C7] `http/client.py::HTTPResponse._read_chunked` http/client.py:591-609 (cited, relevant)
- [C8] `http/client.py::HTTPResponse.read1` http/client.py:655-672

[C5] http/client.py (shown lines near the grader’s citations):
```python
  353              for hdr, val in self.headers.items():
  354                  print("header:", hdr + ":", val)
  355  
  356          # are we using the chunked-style of transfer encoding?
  357          tr_enc = self.headers.get("transfer-encoding")
  358          if tr_enc and tr_enc.lower() == "chunked":
  359              self.chunked = True
  360              self.chunk_left = None
  361          else:
```

**Verdict:** correct
**Justification:** transfer-encoding: chunked sets self.chunked (http/client.py:355-359), read() dispatches to read_chunked (client.py:472-473), which loops over get_chunk_left, appends safe_read data and honours amt (client.py:591-609); close handling in begin/read (client.py:362-394, 463-501)

**Your call:** _(agree / change to …)_

---
