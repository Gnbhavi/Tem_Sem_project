#!/bin/sh
# setup_hooks.sh
# This script installs post-checkout and post-commit hooks into .git/hooks

HOOKS_DIR=".git/hooks"

# Ensure hooks directory exists
if [ ! -d "$HOOKS_DIR" ]; then
  echo "Error: .git/hooks directory not found. Run this inside a Git repo."
  exit 1
fi

# --- Post-checkout hook ---
cat > "$HOOKS_DIR/post-checkout" << 'EOF'
#!/bin/sh
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Only add if it's a weekly branch (name contains 'week')
if echo "$BRANCH" | grep -q "week"; then
    # Check if header already exists
    if ! grep -q "## $BRANCH" CHANGELOG.md 2>/dev/null; then
        echo "\n## $BRANCH" >> CHANGELOG.md
        git add CHANGELOG.md
        git commit -m "Start $BRANCH"
    fi
fi
EOF

chmod +x "$HOOKS_DIR/post-checkout"

# --- Post-commit hook ---
cat > "$HOOKS_DIR/post-commit" << 'EOF'
#!/bin/sh
# Append latest commit message to CHANGELOG.md
BRANCH=$(git rev-parse --abbrev-ref HEAD)

if echo "$BRANCH" | grep -q "week"; then
    echo "- $(git log -1 --pretty=%s)" >> CHANGELOG.md
    git add CHANGELOG.md
    git commit --amend --no-edit
