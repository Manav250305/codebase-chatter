# Render with: vhs docs/demo.tape  (from the repository root)
Output docs/demo.gif

Set Shell "zsh"
Set FontSize 15
Set Width 1400
Set Height 820
Set Padding 20
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"

Hide
Type "export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TQDM_DISABLE=1 TOKENIZERS_PARALLELISM=false"
Enter
Type "export CHATTER_INDEX_DIR=$(mktemp -d)/index PS1='$ '"
Enter
Type "clear"
Enter
Show

Type "chatter index ."
Enter
Sleep 22s

Type `chatter search "how is a quoted evidence line checked against the chunk" -k 5`
Enter
Sleep 8s

Type `chatter ask "How does chatter check that a quoted evidence line really appears in the cited chunk?" -k 5 --prompt plain`
Enter
Sleep 40s
