# mkdir -p /data/combined/{angry,disgusted,fearful,happy,neutral,sad,surprised,unclear}

# Symlink or copy both datasets into one tree
for emotion in angry disgusted fearful happy neutral sad surprised; do
    # Create the target directory if it doesn't exist
    mkdir -p data/combined/"$emotion"

    cp data/ravdess/$emotion/*.mp4  data/combined/$emotion/  2>/dev/null || true
    cp data/cremad/$emotion/*.flv   data/combined/$emotion/  2>/dev/null || true
done
cp data/ravdess/unclear/* data/combined/unclear/ 2>/dev/null || true