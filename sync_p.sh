ULT_PATH="/C/Users/clint/my-files/projects/reddit-sentiment-analysis"

cd "$VAULT_PATH"

CHANGES_EXIST="$(git status --porcelain | wc -l)"

if [ "CHANGES_EXIST" -eq 0 ]; then
	    exit 0
fi

git add .
git commit -q -m "Last Sync: $(date +"%Y-%m-%d %H:%M:%S")"
git push -q
