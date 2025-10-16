#!/bin/bash

# ============================================================================
# REPLACE OLD FILES IN EXISTING GITHUB REPO
# ============================================================================

echo "Step 1: Staging all changes (deletions + additions)..."
git add -A

echo ""
echo "Step 2: Creating commit..."
git commit -m "Major update: Final clean analysis with DML results

CHANGES:
- Updated to final DML analysis (dml_v4_enhanced_v1.R)
- Added comprehensive README with methodology
- Added all result tables and plots (latest_results/)
- Added .gitignore to exclude private data and manuscript
- Removed all old/previous analysis files
- Removed draft manuscripts

RESULTS:
- Manufacturing Value Added: +42.3% (p=0.093)
- Gross Fixed Capital Formation: -95.3% (p=0.047)
- Manufacturing FDI: +82% (p=0.347, not sig)
- Domestic Credit: Excluded (unstable estimates)

NOTE: Dataset (final_final.csv) excluded - available upon request"

echo ""
echo "Step 3: Pushing to GitHub (replacing old files)..."
git push

echo ""
echo "============================================================================"
echo "✅ DONE! Your GitHub repo is now updated with clean final version!"
echo "============================================================================"
echo ""
echo "What was updated:"
echo "  ✅ Code: dml_v4_enhanced_v1.R (final version)"
echo "  ✅ Results: latest_results/ (all tables and plots)"
echo "  ✅ Documentation: README.md (comprehensive)"
echo "  ✅ Privacy: .gitignore (excludes data and manuscript)"
echo "  🗑️  Deleted: All old files, previous attempts, drafts"
echo ""
echo "What is NOT on GitHub (protected by .gitignore):"
echo "  🔒 CLEAN.docx (your manuscript)"
echo "  🔒 final_final.csv (your dataset)"
echo ""

