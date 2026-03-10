# DeepData Lessons

- When a user asks to commit a wide fix set, run the older package-level short suite before calling the branch verified; targeted regression tests are not enough.
- Listener-based Go tests should use an explicit helper instead of raw `httptest.NewServer` so restricted environments skip cleanly rather than panicking on socket creation.
