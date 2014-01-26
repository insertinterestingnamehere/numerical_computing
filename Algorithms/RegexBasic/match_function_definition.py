# Solution to prob:match_function_definition
# The solutions for the previous couple problems are embedded here to; TODO clearly make the solution files
import re

pattern_strings = {
        'id': r"([a-zA-Z_]\w*)",
        'str': r"('[^']*')",
        'num': r"(\d+\.\d*|\.\d+)",
        '_': r"(\s*)"
}
pattern_strings['param_rhs'] = r"(={_}({str}|{num}|{id}))".format(**pattern_strings)
pattern_strings['param'] = r"({id}{_}{param_rhs}?)".format(**pattern_strings)
pattern_strings['param_list'] = r"({param}{_}(,{_}{param})*)".format(**pattern_strings)

pattern_strings['func'] = r"^def {_}{id}{_}\({_}({param_list}{_})?\){_}:$".format(**pattern_strings)
function_pattern = re.compile(pattern_strings['func'])

def test(string):
    return bool(function_pattern.match(string))

def run_tests():
    assert test(r"def compile(pattern,string):")
    assert test(r"def  space  ( ) :")
    assert test(r"def a113(_dir, file_path='\Desktop\files', val=_PI):")
    assert test(r"def func(num=3., num=.5, num=0.0):")
    assert not test(r"def func(num=.):")
    assert not test(r"def do_problem(error, ):")
    assert not test(r"def variable:")
    assert not test(r"def f.f():")
    assert not test(r"def f(*args):")
    assert not test(r"def f(, val):")
    assert not test(r"def f(,):")
    assert not test(r"def err*r(gamma):")
    assert not test(r"def sleep('no parameter name'):")
    assert not test(r"def func(value=_MY_CONSTANT, msg='%s' % _DEFAULT_MSG):")
    assert not test(r"def func(s1='', this one is a little tricky, s2=''):")
    assert not test(r"def func(): Remember your line anchors!")
    assert not test(r"def func()")
    assert not test(r"deffunc():")
    assert not test(r"func():")
    assert not test(r"exit")

    assert test(r"def f( a):")
    assert test(r"def f( a, b):")
    assert test(r"def f(a ):")
    assert test(r"def f(a, b ):")
    assert not test(r"def f(a=3.6f):")
    assert not test(r"def f(a='hug'.4):")
    # assert test(r"")
    print "Passed all tests."
run_tests()

while True:
    input_string = raw_input("Enter a string>>> ")
    if input_string == 'exit':
        break
    print test(input_string) 
