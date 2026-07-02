"""Train a SentencePiece unigram tokenizer for the Balmont/Silver-Age corpus.

Built on the *combined* corpus so pretrain and fine-tune share one vocabulary
(same reason the char vocab was shared). Poetry needs three things a default
SentencePiece setup gets wrong, all handled here:

  * newlines are meaningful (line/stanza structure) -> encoded as a dedicated
    <nl> token; we swap \\n<->'<nl>' around SP so verse layout round-trips.
  * indentation / repeated spaces must survive -> remove_extra_whitespaces off,
    add_dummy_prefix off.
  * poem markers ✦ ✧ must stay atomic -> declared as user-defined symbols.

byte_fallback guarantees any character is representable, retiring the <unk>
hack. Run: python data/build_bpe.py [vocab_size]
"""

import os
import sys
import sentencepiece as spm

HERE = os.path.dirname(__file__)
PRETRAIN = os.path.join(HERE, "russian_silver_age.txt")
FINETUNE = os.path.join(HERE, "tiny_balmont.txt")
PREFIX = os.path.join(HERE, "russian_silver_age_bpe")

NL = "<nl>"  # newline sentinel; a user-defined symbol so it's one atomic token


def to_sp(text: str) -> str:
    return text.replace("\n", NL)


def from_sp(text: str) -> str:
    return text.replace(NL, "\n")


def main():
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

    # One sentence per line, internal newlines -> <nl>, so SP sees reasonable
    # length sentences while <nl> becomes a normal (atomic) token in-stream.
    train_path = os.path.join(HERE, "_bpe_train.txt")
    with open(train_path, "w", encoding="utf-8") as out:
        for src in (PRETRAIN, FINETUNE):
            text = open(src, encoding="utf-8").read()
            for poem in text.split("✦"):
                poem = poem.strip("\n ")
                if poem:
                    out.write("✦" + to_sp("\n" + poem) + "\n")

    spm.SentencePieceTrainer.train(
        input=train_path,
        model_prefix=PREFIX,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        user_defined_symbols=[NL, "✦", "✧"],
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
    )
    os.remove(train_path)

    # Validate a lossless round-trip on a real sample (the acid test).
    sp = spm.SentencePieceProcessor(model_file=PREFIX + ".model")
    sample = open(FINETUNE, encoding="utf-8").read()[2000:2600]
    ids = sp.encode(to_sp(sample))
    back = from_sp(sp.decode(ids))
    print(f"vocab size: {sp.get_piece_size()}")
    print(f"sample: {len(sample)} chars -> {len(ids)} tokens "
          f"({len(sample)/len(ids):.2f} chars/token)")
    print(f"lossless round-trip: {back == sample}")
    if back != sample:
        for i, (a, b) in enumerate(zip(sample, back)):
            if a != b:
                print(f"  first diff at {i}: {sample[i-10:i+10]!r} vs {back[i-10:i+10]!r}")
                break
    print("pieces:", sp.encode(to_sp("✦\nЯ вольный ветер, я вечно вею."), out_type=str))


if __name__ == "__main__":
    main()
