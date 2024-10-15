import os
import sys
import time


class WorkSplitter(object):
    def __init__(self):
        try:
            _, columns = os.popen('stty size', 'r').read().split()
            self.columns = int(columns)
        except:
            self.columns = 50

    def section(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = int(self.columns-name_length-left_length)

        output = '='*self.columns+'\n' \
                 + "|"+' '*(left_length-1)+name+' '*(right_length-1)+'|\n'\
                 + '='*self.columns+'\n'

        print(output)

    def subsection(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = int(self.columns-name_length-left_length)

        output = '#' * (left_length-1) + ' ' + name + ' ' + '#' * (right_length-1) + '\n'
        print(output)

    def subsubsection(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = self.columns-name_length-left_length

        output = '-' * (left_length-1) + ' ' + name + ' ' + '-' * (right_length-1) + '\n'
        print(output)


def inhour(elapsed):
    return time.strftime('%H:%M:%S', time.gmtime(elapsed))


def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()
