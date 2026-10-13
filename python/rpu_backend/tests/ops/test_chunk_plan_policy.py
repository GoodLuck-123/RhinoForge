import sys
import rpu_backend
from rpu_backend._cpp_ext import chunk_helpers as c
FAILED = 0

def check(cond, msg):
    global FAILED
    if cond:
        print(f'  ok   {msg}')
    else:
        FAILED += 1
        print(f'  FAIL {msg}')

def chunks(seq_len, cs):
    return (seq_len + cs - 1) // cs

def deficit(seq_len, cs):
    return chunks(seq_len, cs) * cs - seq_len

def brute_force(seq_len, candidates):
    return min(candidates, key=lambda cs: (chunks(seq_len, cs), deficit(seq_len, cs)))

def main() -> int:
    aligned = [cs for cs in range(16, 1041, 16)]
    print('[1] fewer chunks always wins, even at a much worse tail')
    check(c.best_chunk_size(576, [512, 448, 384, 320, 256]) == 320, 'seq=576 picks 320 (2 chunks, deficit 64) over 512 (2, deficit 448)')
    check(not c.prefer_balanced(576, 192, 320), 'seq=576 rejects 192 despite deficit 0 — it costs a third chunk')
    print('[2] the rule matches an independent implementation, incl. odd lengths')
    for seq_len in (1, 15, 16, 17, 63, 100, 128, 313, 576, 700, 999, 1000, 1023, 1024, 1040):
        got = c.best_chunk_size(seq_len, aligned)
        want = brute_force(seq_len, aligned)
        check(got == want, f'seq={seq_len}: {got} == {want} (n={chunks(seq_len, got)}, deficit={deficit(seq_len, got)})')
    print('[3] the winner is always the most even split of its chunk count')
    for seq_len in (313, 700, 999, 1000, 1023):
        best = c.best_chunk_size(seq_len, aligned)
        n = chunks(seq_len, best)
        rivals = [cs for cs in aligned if chunks(seq_len, cs) == n]
        check(all((deficit(seq_len, best) <= deficit(seq_len, cs) for cs in rivals)), f'seq={seq_len}: cs={best} has the smallest deficit among the {len(rivals)} candidates that also give {n} chunk(s)')
    print('[4] a restricted candidate set (SPM / predicate rejects) still applies the same order')
    check(c.best_chunk_size(1024, [512, 256, 128]) == 512, 'seq=1024 restricted to {512,256,128} picks 512 (2 chunks)')
    check(c.best_chunk_size(1000, [512, 256, 128]) == 512, 'seq=1000 restricted picks 512 (2 chunks, deficit 24)')
    check(c.best_chunk_size(1000, [256, 128]) == 256, 'without 512, seq=1000 takes 256 (4 chunks) over 128 (8)')
    print('[5] empty history accepts anything; the rule is a strict improvement')
    check(c.prefer_balanced(1000, 512, 0), "any candidate beats 'nothing yet'")
    check(not c.prefer_balanced(1000, 512, 512), 'a tie is not an improvement')
    print(f"\n{('PASS' if FAILED == 0 else 'FAIL')}: {FAILED} failed")
    return 1 if FAILED else 0
if __name__ == '__main__':
    sys.exit(main())

def test_native_selection_contract():
    global FAILED
    FAILED = 0
    assert main() == 0
