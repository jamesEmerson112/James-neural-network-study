"""
Turing Machine Simulator
========================
A complete, runnable implementation with three examples:
1. Binary increment
2. Palindrome checker
3. Busy Beaver (3-state)

Run: python turing_machine_simulator.py
"""


class TuringMachine:
    """
    A complete Turing Machine simulator.

    transitions: dict mapping (state, symbol) -> (new_state, write_symbol, direction)
    direction: 'L' for left, 'R' for right
    """

    def __init__(self, transitions, start_state, accept_states, reject_states=None, blank='⊔'):
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
        self.reject_states = reject_states or set()
        self.blank = blank

    def run(self, input_string, max_steps=10000, verbose=False):
        # Initialize tape as a dict (sparse representation of infinite tape)
        tape = {}
        for i, symbol in enumerate(input_string):
            tape[i] = symbol

        head = 0
        state = self.start_state
        steps = 0

        while steps < max_steps:
            # Read symbol under head (blank if empty)
            symbol = tape.get(head, self.blank)

            if verbose:
                self._print_tape(tape, head, state, steps)

            # Check for halt
            if state in self.accept_states:
                return self._read_tape(tape), True, steps
            if state in self.reject_states:
                return self._read_tape(tape), False, steps

            # Look up transition
            key = (state, symbol)
            if key not in self.transitions:
                return self._read_tape(tape), False, steps  # No transition = reject

            new_state, write_symbol, direction = self.transitions[key]

            # Write, move, change state
            tape[head] = write_symbol
            head += 1 if direction == 'R' else -1
            state = new_state
            steps += 1

        return self._read_tape(tape), None, steps  # None = didn't halt (timeout)

    def _read_tape(self, tape):
        """Read non-blank contents of the tape."""
        if not tape:
            return ''
        min_pos = min(tape.keys())
        max_pos = max(tape.keys())
        result = ''
        for i in range(min_pos, max_pos + 1):
            sym = tape.get(i, self.blank)
            if sym != self.blank:
                result += sym
        return result

    def _print_tape(self, tape, head, state, step):
        """Visualize the current tape configuration."""
        if not tape and head == 0:
            print(f"  Step {step:3d}  State: {state:10s}  [{self.blank}]")
            return
        positions = set(tape.keys()) | {head}
        min_pos = min(positions) - 1
        max_pos = max(positions) + 1
        tape_str = ''
        for i in range(min_pos, max_pos + 1):
            sym = tape.get(i, self.blank)
            if i == head:
                tape_str += f'[{sym}]'
            else:
                tape_str += f' {sym} '
        print(f"  Step {step:3d}  State: {state:10s}  {tape_str}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 1: Binary Increment
# ─────────────────────────────────────────────────────────────

def demo_binary_increment():
    print("=" * 60)
    print("EXAMPLE 1: Binary Increment (adds 1 to a binary number)")
    print("=" * 60)

    transitions = {
        # Go right to find the end
        ('start', '0'): ('start', '0', 'R'),
        ('start', '1'): ('start', '1', 'R'),
        ('start', '⊔'): ('add',   '⊔', 'L'),

        # Add 1 (propagate carry from right to left)
        ('add', '0'):   ('done',  '1', 'L'),
        ('add', '1'):   ('add',   '0', 'L'),
        ('add', '⊔'):   ('done',  '1', 'R'),

        # Go right to end, then halt
        ('done', '0'):  ('done',  '0', 'R'),
        ('done', '1'):  ('done',  '1', 'R'),
        ('done', '⊔'):  ('halt',  '⊔', 'L'),
    }

    tm = TuringMachine(
        transitions=transitions,
        start_state='start',
        accept_states={'halt'}
    )

    test_cases = [
        ('1011', 11, 12),
        ('111',   7,  8),
        ('1001',  9, 10),
        ('0',     0,  1),
    ]

    for binary, decimal_in, decimal_out in test_cases:
        print(f"\n  Input: {binary} (decimal {decimal_in})")
        result, accepted, steps = tm.run(binary, verbose=True)
        print(f"  Result: {result} (decimal {decimal_out}) — {steps} steps\n")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 2: Palindrome Checker
# ─────────────────────────────────────────────────────────────

def demo_palindrome():
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Palindrome Checker")
    print("=" * 60)
    print("  Strategy: repeatedly match & erase first and last chars\n")

    transitions = {
        # Check first character
        ('q0', '0'): ('have0', '⊔', 'R'),
        ('q0', '1'): ('have1', '⊔', 'R'),
        ('q0', '⊔'): ('accept', '⊔', 'R'),

        # Carrying '0' — go right to find last character
        ('have0', '0'): ('have0', '0', 'R'),
        ('have0', '1'): ('have0', '1', 'R'),
        ('have0', '⊔'): ('check0', '⊔', 'L'),

        # Check if last char matches '0'
        ('check0', '0'): ('back', '⊔', 'L'),
        ('check0', '1'): ('reject', '1', 'R'),
        ('check0', '⊔'): ('accept', '⊔', 'R'),

        # Carrying '1' — go right to find last character
        ('have1', '0'): ('have1', '0', 'R'),
        ('have1', '1'): ('have1', '1', 'R'),
        ('have1', '⊔'): ('check1', '⊔', 'L'),

        # Check if last char matches '1'
        ('check1', '1'): ('back', '⊔', 'L'),
        ('check1', '0'): ('reject', '0', 'R'),
        ('check1', '⊔'): ('accept', '⊔', 'R'),

        # Go back to the start
        ('back', '0'): ('back', '0', 'L'),
        ('back', '1'): ('back', '1', 'L'),
        ('back', '⊔'): ('q0',   '⊔', 'R'),
    }

    tm = TuringMachine(
        transitions=transitions,
        start_state='q0',
        accept_states={'accept'},
        reject_states={'reject'}
    )

    test_cases = ['1001', '1010', '10101', '110', '1', '']
    for s in test_cases:
        result, accepted, steps = tm.run(s)
        status = "PALINDROME" if accepted else "NOT PALINDROME"
        display = f"'{s}'" if s else "''"
        print(f"  {display:10s} → {status:16s} ({steps} steps)")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 3: Busy Beaver (3-state)
# ─────────────────────────────────────────────────────────────

def demo_busy_beaver():
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Busy Beaver (3-state)")
    print("=" * 60)
    print("  The most productive 3-state machine possible.")
    print("  Writes the maximum number of 1s before halting.\n")

    transitions = {
        ('A', '0'): ('B', '1', 'R'),
        ('A', '1'): ('C', '1', 'L'),
        ('B', '0'): ('A', '1', 'L'),
        ('B', '1'): ('B', '1', 'R'),
        ('C', '0'): ('B', '1', 'L'),
        ('C', '1'): ('halt', '1', 'R'),
    }

    tm = TuringMachine(
        transitions=transitions,
        start_state='A',
        accept_states={'halt'},
        blank='0'
    )

    result, accepted, steps = tm.run('', verbose=True, max_steps=100)
    ones = result.count('1')
    print(f"\n  Busy Beaver wrote {ones} ones in {steps} steps")
    print(f"  (This is the proven maximum for 3 states)")
    print()
    print("  Known Busy Beaver values:")
    print("  ┌────────┬───────┬───────┐")
    print("  │ States │  1s   │ Steps │")
    print("  ├────────┼───────┼───────┤")
    print("  │   1    │    1  │     1 │")
    print("  │   2    │    4  │     6 │")
    print("  │   3    │    6  │    14 │")
    print("  │   4    │   13  │   107 │")
    print("  │   5    │ 4098  │ 47176 │")
    print("  │   6    │  ???  │  ???  │  ← unknown, possibly incomputable")
    print("  └────────┴───────┴───────┘")


# ─────────────────────────────────────────────────────────────
# BONUS: The Halting Problem (demonstration of the paradox)
# ─────────────────────────────────────────────────────────────

def demo_halting_problem():
    print("\n" + "=" * 60)
    print("BONUS: The Halting Problem (why it's unsolvable)")
    print("=" * 60)
    print("""
  Suppose we COULD write a function:

    def halts(program, input):
        \"\"\"Returns True if program(input) halts, False otherwise.\"\"\"
        ...  # magic oracle

  Now construct a paradox:

    def paradox(x):
        if halts(paradox, x):   # If I would halt...
            while True: pass    # ...then loop forever
        else:                   # If I would loop...
            return              # ...then halt

  Call paradox(0):
    - If halts() says True  → paradox loops   → halts() was WRONG
    - If halts() says False → paradox returns  → halts() was WRONG

  CONTRADICTION. Therefore halts() CANNOT EXIST.

  This is Turing's proof that some problems are fundamentally
  undecidable — no computer can ever solve them.
""")


if __name__ == '__main__':
    demo_binary_increment()
    demo_palindrome()
    demo_busy_beaver()
    demo_halting_problem()
