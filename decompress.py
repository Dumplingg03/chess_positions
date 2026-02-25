import zstandard as zstd

input_path = "lichess_db_puzzle.csv.zst"
output_path = "lichess_puzzles.csv"

with open(input_path, "rb") as compressed:
    dctx = zstd.ZstdDecompressor()
    with open(output_path, "wb") as decompressed:
        dctx.copy_stream(compressed, decompressed)

print("Готово! CSV распакован.")