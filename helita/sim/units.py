"""
Created by Sam Evans on Apr 27 2021

purpose: enabling "units" mode for DataClass objects (e.g. BifrostData, EbysusData).

The idea is to:
- have all load_quantities functions return values in simulation units.
- have a way to lookup how to convert any variable to a desired set of units.
- have an attribute of the DataClass object tell which system of units we want output.

FOR NOW:
>>>> Use obj.get_units() to see the units for the most-recent quantity. (at top of QUANTS_TREE.) <<<< 

State of the code right now:
- The "hesitant execution" of methods in here means that if you do not call obj.get_units()
    or any other units-related functions, then nothing in units.py should cause a crash.
- There are very few units actually implemented in load_..._quantities files.
    - This implementation is pending a possible addition of some "quality-of-life" improvements:
        - e.g. "universal naming scheme" (see TODO)
        - e.g. "common unit tuples" (see TODO)
    - The units which are actually implemented are:
        - units in get_deriv in load_arithmetic_quantities
        - units in get_center in load_arithmetic_quantities
        - units in get_square in load_arithmetic_quantities
        - units for 'nr' and 'uix' in get_onefluid_var in load_mf_quantities

TODO:
- (high priority) make a "User-friendly" guide to units.py at the top of this file.
    - (in the meantime you can try to figure it out from examples in load_..._quantities files)
- have a units_system flag attribute which allows to convert units at top level automatically.
    - (By default the conversion will be off.)
    - Don't tell Juan about this attribute because he won't like it ;) but he doesn't ever have to use it!
- create a "universal naming scheme" which allows to make unit symbols "universal", e.g.
    a unit symbol, U_L which will be 'cm' for cgs and 'm' for si.
- (maybe) create "common unit tuples" e.g. Lenth_Tuple = UnitsTuple(USI.l, U_SYM('m'))
- test that the code is working properly...


"""

# import built-ins
import operator
import functools
import collections

''' ----------------------------- Set Defaults ----------------------------- '''

# whether to hide tracebacks from internal funcs in this file when showing error traceback.
HIDE_INTERNAL_TRACEBACKS = True

# for ATF = AttrsFunclike(..., format_attr=None, **kw__entered),
## if kw__entered[ATTR_FORMAT_KWARG] (:=kw_fmt) exists,
### for any required kwargs which haven't been entered,
#### try to use the attribute of obj: kw_fmt(kwarg).
#### (Instead of trying to use the attribute of obj: kwarg.)
ATTR_FORMAT_KWARG = '_attr_format'

# when doing any of the quant_child methods in UnitsFuncBuilder,
# use UNITS_KEY_KWARG to set units_key, unless self.units_key is set (e.g. at initialization).
## This affects UNI (for which self.units_key is None), but not USI nor UCGS.
UNITS_KEY_KWARG = '_units_key'

# UNITS_MODES stores the internal ("behind-the-scenes") info for unit conversion modes.
#   units_key = key in which UnitsTuple is stored in vardict;
#               units_tuple = obj.vardict[metaquant][typequant][quant][units_key]
#   attr_format = value to pass to ATTR_FORMAT_KWARG. (See ATTR_FORMAT_KWARG above for more info.)
UNITS_MODES = \
{
    # mode : (units_key, attr_format)
    'si'   : ('usi', 'usi_{}'),
    'cgs'  : ('ucgs', 'u_{}'),
}

# UNITS_UNIVERSAL_KEY is a key which allows to use only one UnitsTuple to represent multiple unit systems.
# In vardict when searching for original units_key, if it is not found, we will also search for this key;
# i.e. if vardict[metaquant][typequant][quant][UNITS_UNIVERSAL_KEY] (:= this_unit_tuple) exists,
## then we will call this_unit_tuple(obj.uni, obj, ATTR_FORMAT_KWARG = units_key)
# For example, a velocity is represented by obj.uni.usi_u or obj.uni.u_u, but these are very similar.
# So, instead of setting usi and ucgs separately for docvar('uix'), we can just set uni:
# docvar('uix', 'x-component of ifluid velocity', uni = UNI.l)
UNITS_UNIVERSAL_KEY = 'uni'

# UNITS_F_KEY
UNITS_KEY_F = '{}_f'

# UNITS_NAME_KEY
UNITS_KEY_NAME = '{}_name'


''' ----------------------------- Hesitant Execution ----------------------------- '''
# in this section, establish objects which can be combined like math terms but
# create a function which can be evaluated later, instead of requiring evaluation right away.
# See examples in FuncBuilder documentation.

class FuncBuilder:
    '''use this object to build a function one arg / kwarg at a time.
    
    use attributes for kwargs, indices for args.

    Examples:
        u = FuncBuilder()
        f = ((u.x + u.r) * u[1])   # f is a function which does: return (kwarg['x'] + kwarg['r']) * arg[1]
        f(None, 10, x=3, r=5)
        >>> 80
        f = u[0] + u[1] + u[2]     # f is a function which adds the first three args and returns the result.
        f(2, 3, 4)
        >>> 9
        f = u[0] ** u[1]           # f(a,b) is equivalent to a**b
        f(7,2)
        >>> 49

    Technically, this object returns Funclike objects, not functions.
    That means you can combine different FuncBuilder results into a single function.
    Example:
        u = FuncBuilder()
        f1 = (u.x + u.r) * u[1]
        f2 = u.y + u[0] + u[1]
        f = f1 - f2
        f(0.1, 10, x=3, r=5, y=37)
        >>> 32.9            # ((3 + 5) * 10) - (37 + 0.1 + 10)
    '''
    def __init__(self, FunclikeType=None, **kw__funclike_init):
        '''convert functions to funcliketype object.'''
        self.FunclikeType = Funclike if (FunclikeType is None) else FunclikeType
        self._kw__funclike_init = kw__funclike_init

    def __getattr__(self, a):
        '''returns f(*args, **kwargs) --> kwargs[a].'''
        def f_a(*args, **kwargs):
            '''returns kwargs[{a}]'''
            try:
                return kwargs[a]
            except KeyError as e:
                message = 'Expected kwargs to contain key {} but they did not!'.format(a)
                raise KeyError(message) from None
        f_a.__doc__  = f_a.__doc__.replace('{a}', repr(a))
        f_a.__name__ = a
        return self.FunclikeType(f_a, required_kwargs=[a], **self._kw__funclike_init)
    
    def __getitem__(self, i):
        '''returns f(*args, **kwargs) --> args[i]. i must be an integer.'''
        def f_i(*args, **kwargs):
            '''returns args[{i}]'''
            try:
                return args[i]
            except IndexError:
                raise IndexError('Expected args[{}] to exist but it did not!'.format(i))
        f_i.__doc__  = f_i.__doc__.replace('{i}', repr(i))
        f_i.__name__ = 'arg' + str(i)
        return self.FunclikeType(f_i, required_args=[i], **self._kw__funclike_init)
    
def make_Funclike_magic(op, op_name=None):
    '''makes magic funclike for binary operator
    it will be named magic + op_name.
    
    Example:
        f = make_Funclike_magic(operator.__mul__, '__times__')
        >>> a function named magic__times__ which returns a Funclike object that does a * b.
    
    make_Funclike_magic is a low-level function which serves as a helper function for the Funclike class.
        
    make_Funclike_magic returns a function of (a, b) which returns a Funclike-like object that does op(a, b).
    type(result) will be type(a) unless issubclass(b, a), in which case it will be type(b).
    '''
    def magic(a, b):
        typeA, typeB = type(a), type(b)
        # apply operation
        if callable(b):
            def f(*args, **kwargs):
                __tracebackhide__ = HIDE_INTERNAL_TRACEBACKS
                return op(a(*args, **kwargs), b(*args, **kwargs))
            bname = getattr(b, '__name__', typeB.__name__)
        else:
            def f(*args, **kwargs):
                __tracebackhide__ = HIDE_INTERNAL_TRACEBACKS
                return op(a(*args, **kwargs), b)
            bname = str(b)
            if len(bname) > 5:
                bname = 'value'
        # set name of f
        if op_name is not None:
            f.__name__ = '(' + a.__name__ + op_name + bname + ')'
        # typecast f to appropriate type of Funclike-like object, and return it.
        if issubclass(typeB, typeA):
            returntype = typeB
        else:
            returntype = typeA
        return returntype(f, parents=[a,b])
    if op_name is not None:
        magic.__name__ = 'magic' + op_name
    return magic
    
class Funclike:
    '''function-like object. Useful for combining with other Funclike objects.
    Allows for "hesitant execution":
        The args and kwargs do not need to be known until later.
        Evaluate whenever the instance is called like a function.

    Example:
        # --- basic example ---
        getx           = lambda *args, **kwargs: kwargs['x']
        funclike_getx  = Funclike(getx)
        mult_x_by_2    = funclike_getx * 2
        mult_x_by_2(x=7)
        >>> 14      # 7 * 2
        # --- another basic example ---
        gety           = lambda *args, **kwargs: kwargs['y']
        funclike_gety  = Funclike(gety)
        get0           = lambda *args, **kwargs: args[0]
        funclike_get0  = Funclike(get0)
        add_arg0_to_y  = funclike_get0 + funclike_gety
        add_arg0_to_y(3, y=10) 
        >>> 13      # 3 + 10    
        # --- combine the basic examples ---
        add_arg0_to_y_then_subtract_2x = add_arg0_to_y - mult_x_by_2
        add_arg0_to_y_then_subtract_2x(7, y=8, x=50)
        >>> -85     # (7 + 8) - 50 * 2
    '''
    def __init__(self, f, required_args=[], required_kwargs=[], parents=[]):
        self._tracebackhide = HIDE_INTERNAL_TRACEBACKS
        self.f = f
        self.__name__ = f.__name__
        self._required_args   = required_args     # list of  args  which must be provided for a function call to self.
        self._required_kwargs = required_kwargs   # list of kwargs which must be provided for a function call to self.
        for parent in parents:
            parent_req_args   = getattr(parent, '_required_args', [])
            parent_req_kwargs = getattr(parent, '_required_kwargs', [])
            if len(parent_req_args)>0:
                self._add_to_required('_required_args', parent_req_args)
            if len(parent_req_kwargs)>0:
                self._add_to_required('_required_kwargs', parent_req_kwargs)
        
    def _add_to_required(self, original_required, new_required):
        orig = getattr(self, original_required, [])
        setattr(self, original_required,  sorted(list( set(orig + new_required) ))  )

    def __call__(self, *args, **kwargs):
        __tracebackhide__ = self._tracebackhide
        return self.f(*args, **kwargs)
        
    __mul__     = make_Funclike_magic(operator.__mul__,     ' * ')    # multiply
    __add__     = make_Funclike_magic(operator.__add__,     ' + ')    # add
    __sub__     = make_Funclike_magic(operator.__sub__,     ' - ')    # subtract
    __truediv__ = make_Funclike_magic(operator.__truediv__, ' / ')    # divide
    __pow__     = make_Funclike_magic(operator.__pow__,     ' ** ')   # raise to a power

    def _strinfo(self):
        '''info about self. (goes to repr)'''
        return 'required_args={}, required_kwargs={}'.format(
                    self._required_args, self._required_kwargs)

    def __repr__(self):
        return '<{} named {} with {}>'.format(
                object.__repr__(self), repr(self.__name__), self._strinfo())


class AttrsFunclike(Funclike):
    '''Funclike but treat args[argn] as obj, and use obj attrs for un-entered required kwargs.

    argn: int (default 0)
        treat args[argn] as obj.
    format_attr: string, or None.
        string --> format this string to all required kwargs before checking if they are attrs of obj.
                   E.g. format = 'usi_{}' --> if looking for 'r', check obj.usi_r and kwargs['r'].
        None --> by default, don't mess with kwargs names at all.
                 However, if special kwarg ATTR_FORMAT_KWARG (defined at top of units.py)
                 is passed to this function, use its value to format the required kwargs.
    '''
    def __init__(self, f, argn=0, format_attr=None, **kw__funclike_init):
        '''f should be a Funclike object.'''
        Funclike.__init__(self, f, **kw__funclike_init)
        self.argn=argn
        self.format_attr=format_attr
        self._add_to_required('_required_args', [argn])
        self._add_to_required('_required_args_special', [argn])
        f               = self.f
        required_kwargs = self._required_kwargs
        def f_attrs(*args, **kwargs):
            __tracebackhide__ = self._tracebackhide
            obj     = args[argn]
            kwdict  = kwargs
            if self.format_attr is None:
                format_attr = kwargs.get(ATTR_FORMAT_KWARG, '{}')
            else:
                format_attr = self.format_attr
            # for any required kwargs which haven't been entered,
            ## try to use the attribute (format_attr.format(kwarg)) of obj, if possible.
            for kwarg in required_kwargs:
                if kwarg not in kwargs:
                    attr_name = format_attr.format(kwarg)
                    if hasattr(obj, attr_name):
                        kwdict[kwarg] = getattr(obj, attr_name)
            return f(*args, **kwdict)
        f_attrs.__name__ = f.__name__    # TODO: add something to the name to indicate it is an AttrsFunclike.
        Funclike.__init__(self, f_attrs, self._required_args, self._required_kwargs)

    # TODO: would it be cleaner to keep the original f and just override the __call__ method?
    # Doing so may allow to move the args & kwargs checking to __call__,
    # i.e. check that we have required_args & required_kwargs, else raise error.

    def _special_args_info(self):
        '''info about the meaning of special args for self.'''
        return 'arg[{argn}] attrs can replace missing required kwargs.'.format(argn=self.argn)

    def _strinfo(self):
        '''info about self. (goes to repr)'''
        return 'required_args={}, required_kwargs={}. special_args={}: {}'.format(
                    self._required_args, self._required_kwargs,
                    self._required_args_special, self._special_args_info())


''' -------------------------------------------------------------------------- '''
''' ----------------------------- Units-Specific ----------------------------- '''
''' -------------------------------------------------------------------------- '''
# Up until this point in the file, the codes have been pretty generic.
# The codes beyond this section, though, are specific to the units implementation in helita.

''' ----------------------------- Units Naming ----------------------------- '''

# string manipulation helper functions
def _pretty_str_unit(name, value, flip=False, flip_neg=False):
    '''returns string for name, value. name is name of unit; value is exponent.
    flip --> pretend result is showing up in denominator (i.e. multiply exponent by -1).
    flip_neg --> flip only negative values. (Always give a positive exponent result.)

    return (string, whether it was flipped).
    '''
    flipped = False
    if value == 0:
        result = ''
    elif flip or ((value < 0) and flip_neg):
        result = _pretty_str_unit(name, -1 * value, flip=False)[0]
        flipped = True
    elif value == 1:
        result = name
    else:
        result = name + '^{' + str(value) + '}'
    return (result, flipped)

def _join_strs(strs, sep=' '):
        '''joins strings, separating by sep. Ignores Nones and strings of length 0.'''
        ss = [s for s in strs if (s is not None) and (len(s)>0)]
        return sep.join(ss)


class UnitsExpression:
    '''expression of units.

    Parameters
    ----------
    contents: dict
        keys = unit name; values = exponent for that unit.
    order: string (default 'entered')
        determines order in which units are printed. Options are:
            'entered' --> use the order in which the keys appear in contents.
            'exp'     --> order by exponent (descending by default, i.e. largest first).
            'absexp'  --> order by abs(exponent) (decending by default).
            'alpha'   --> order alphabetically (a to z by default).

    TODO: make ordering options "clearer" (use enum or something like that?)

    TODO: make display mode options (e.g. "latex", "pythonic", etc)
    '''
    def __init__(self, contents=collections.OrderedDict(), order='entered', frac=True):
        self.contents = contents
        self.order    = order
        self.frac     = frac      # whether to show negatives in denominator

    def _order_exponent(self, ascending=False):
        '''returns keys for self.contents, ordered by exponent.
        not ascending --> largest first; ascending --> largest last.
        '''
        return sorted(list(self.contents.keys()),
                      key=lambda k: self.contents[k], reverse=not ascending)

    def _order_abs_exponent(self, ascending=False):
        '''returns keys for self.contents, ordered by |exponent|.
        not ascending --> largest first; ascending --> largest last.
        '''
        return sorted(list(self.contents.keys()),
                      key=lambda k: abs(self.contents[k]), reverse=not ascending)

    def _order_alphabetical(self, reverse=False):
        '''returns keys for self.contents in alphabetical order.
        not reverse --> a first; reverse --> a last.
        '''
        #TODO: handle case of '$' included in key name (e.g. for greek letters)
        return sorted(list(self.contents.keys()), reverse=reverse)

    def _order_entered(self, reverse=False):
        '''returns keys for self.contents in order entered.
        reverse: whether to reverse the order.
        '''
        return list(self.contents.keys())

    def _pretty_str_key(self, key, flip=False, flip_neg=False):
        '''determine string given key (a unit's name).
        flip --> always flip; i.e. multiply value by -1.
        flip_neg --> only flip negative values.

        returns (string, whether it was flipped)
        '''
        return _pretty_str_unit(key, self.contents[key], flip=flip, flip_neg=flip_neg)

    def __str__(self):
        '''str of self: pretty string telling the units which self represents.'''
        if self.order == 'exp':
            key_order = self._order_exponent()
        elif self.order == 'absexp':
            key_order = self._order_abs_exponent()
        elif self.order == 'alpha':
            key_order = self._order_alphabetical()
        elif self.order == 'entered':
            key_order = self._order_entered()
        else:
            errmsg = ("self.order is not a valid order! For valid choices,"
                      "see help(UnitsExpression). (really helita.sim.units.UnitsExpression)")
            raise ValueError(errmsg)
        x = [self._pretty_str_key(key, flip_neg=self.frac) for key in key_order]
        numer = [s for s, flipped in x if not flipped]
        denom = [s for s, flipped in x   if   flipped] # and s != ''
        numer_str = _join_strs(numer, ' ')
        if len(denom) == 0:
            result = numer_str
        else:
            if len(numer) == 0:
                numer_str = '1'
            if len(denom) == 1:
                result = numer_str + ' / ' + denom[0]
            else:
                denom_str = _join_strs(denom, ' ')
                result = numer_str + ' / (' + denom_str + ')'
        return result

    def __repr__(self):
        '''repr of self.'''
        return "<{} with content = '{}'>".format(object.__repr__(self), str(self))

    def __mul__(self, b):
        '''multiplication with b (another UnitsExpression object).'''
        result = self.contents.copy()
        if not isinstance(b, UnitsExpression):
            raise TypeError('Expected UnitsExpression type but got type={}'.format(type(b)))
        for key, val in b.contents.items():
            try:
                result[key] += val
            except KeyError:
                result[key] = val
        return UnitsExpression(result, order=self.order, frac=self.frac)

    def __truediv__(self, b):
        '''division by b (another UnitsExpression object).'''
        result = self.contents.copy()
        if not isinstance(b, UnitsExpression):
            raise TypeError('Expected UnitsExpression type but got type={}'.format(type(b)))
        for key, val in b.contents.items():
            try:
                result[key] -= val
            except KeyError:
                result[key] = -1 * val
        return UnitsExpression(result, order=self.order, frac=self.frac)

    def __pow__(self, b):
        '''raising to b (a number).'''
        result = self.contents.copy()
        for key in result.keys():
            result[key] *= b
        return UnitsExpression(result, order=self.order, frac=self.frac)
            

class UnitSymbol(UnitsExpression):
    '''symbol for a single unit.

    UnitSymbol('x') is like UnitsExpression(contents=collections.OrderedDict(x=1))

    Example:
        for 'V^{2} / m', one would enter:
            result = units.UnitSymbol('V')**2 / units.UnitSymbol('m')
        to set printout settings, attributes can be editted directly:
            result.order = 'exp'
            result.frac = True
        to see contents, convert to string:
            str(result)
            >>> 'V^{2} / m'
    '''
    def __init__(self, name, *args, **kwargs):
        self.name = name
        contents = collections.OrderedDict()
        contents[name] = 1
        UnitsExpression.__init__(self, contents, *args, **kwargs)

UnitsSymbol = UnitSymbol    # alias

def UnitSymbols(names, *args, **kwargs):
    '''returns UnitSymbol(name, *args, **kwargs) for name in names.
    names can be a string or list:
        string --> treat names as names.split()
        list   --> treat names list of names.

    Example:
        V, m, s = units.UnitSymbols('V m s', order='absexp')
        str(V**2 / s * m**-4)
        >>> 'V^{2} / (m^{4} s)'
    '''
    if isinstance(names, str):
        names = names.split()
    return tuple(UnitSymbol(name, *args, **kwargs) for name in names)

UnitsSymbols = UnitSymbols  # alias


class UnitsNotFoundError(Exception):
    '''base class for telling that units have not been found.'''
    pass

def _default_units_f(info=''):
    def f(*args, **kwargs):
        errmsg = ("Cannot calculate units. Either the original quant's units are unknown,"
                   " or one of the required children quants' units are unknown.\n"
                   "Further info provided: " + str(info))
        raise UnitsNotFoundError(errmsg)
    return Funclike(f)

DEFAULT_UNITS_F = _default_units_f()
DEFAULT_UNITS_NAME = UnitsExpression()

''' ----------------------------- Units Tuple ----------------------------- '''

UnitsTupleBase = collections.namedtuple('Units', ('f', 'name'),
                                        defaults=[DEFAULT_UNITS_F, DEFAULT_UNITS_NAME]
                                        )

def make_UnitsTuple_magic(op, op_name=None):
    '''makes magic func for binary operator acting on UnitsTuple object.
    it will be named magic + op_name.
    
    make_UnitsTuple_magic is a low-level function which serves as a helper function for the UnitsTuple class.
    '''
    def magic(a, b):
        # apply operation
        if isinstance(b, UnitsTuple):
            f    = op(a.f, b.f)
            name = op(a.name, b.name)
        else:
            f    = op(a.f, b)
            name = op(a.name, b)
        return UnitsTuple(f, name)
    # rename magic (based on op_name)
    if op_name is not None:
        magic.__name__ = 'magic' + op_name
    return magic

class UnitsTuple(UnitsTupleBase):
    '''UnitsTuple tells:
        f: Funclike (or constant...). Call this to convert to the correct units.
        name: UnitsExpression object which gives name for units, e.g. str(UnitsTuple().name)

    Additionally, multiplying, dividing, or exponentiating with another UnitsTuple works intuitively:
        op(UnitsTuple(a1,b1), UnitsTuple(a2,b2)) = UnitsTuple(op(a1,a2), op(b1,b2)) for op in *, /, **.
    And if the second object is not a UnitsTuple, the operation is distributed instead:
        op(UnitsTuple(a1,b1), x) = UnitsTuple(op(a1,x), op(b1,x)) for op in *, /, **.
    '''
    __mul__     = make_UnitsTuple_magic(operator.__mul__,     ' * ')    # multiply
    __add__     = make_UnitsTuple_magic(operator.__add__,     ' + ')    # add
    __sub__     = make_UnitsTuple_magic(operator.__sub__,     ' - ')    # subtract
    __truediv__ = make_UnitsTuple_magic(operator.__truediv__, ' / ')    # divide
    __pow__     = make_UnitsTuple_magic(operator.__pow__,     ' ** ')   # raise to a power

    def __call__(self, *args, **kwargs):
        return UnitsTuple(self.f(*args, **kwargs), self.name)

''' ----------------------------- Dimensionless Tuple ----------------------------- '''
# in this section is a units tuple which should be used for dimensionless quantities.

def dimensionless_units_f(*args, **kwargs):
    '''returns 1, regardless of args and kwargs.'''
    return 1

DIMENSIONLESS_UNITS = Funclike(dimensionless_units_f)
NO_UNITS = DIMENSIONLESS_UNITS

DIMENSIONLESS_NAME  = UnitsExpression()
NO_NAME  = DIMENSIONLESS_NAME

DIMENSIONLESS_TUPLE = UnitsTuple(DIMENSIONLESS_UNITS, DIMENSIONLESS_NAME)


''' ----------------------------- Units FuncBuilder ----------------------------- '''

class UnitsFuncBuilder(FuncBuilder):
    '''FuncBuilder but also qc attribute will get quant children of obj.
    FunclikeType must be (or be a subclass of) AttrsFunclike.
    '''
    def __init__(self, FunclikeType=AttrsFunclike, units_key=None, **kw__funclike_init):
        FuncBuilder.__init__(self, FunclikeType=FunclikeType, **kw__funclike_init)
        self.units_key = units_key

    def _quant_child(self, i, oldest_first=True, return_type='tuple'):
        '''returns a Funclike which gets i'th quant child in QUANTS_TREE for object=args[1].
        
        not intended to be called directly; instead use alternate functions as described below.

        return_type: string (default 'tuple')
            'tuple' --> return a UnitsTuple object.
                        (alternate funcs: quant_child, qc)
            'units' --> return the units function only. (UnitsTuple.f)
                        (alternate funcs: quant_child_units, qcu)
            'name'  --> return the units name only. (UnitsTuple.name)
                        (alternate funcs: quant_child_name, qcn)
        '''
        return_type = return_type.lower()
        assert return_type in ('tuple', 'units', 'name'), 'Got invalid return_type(={})'.format(repr(return_type))
        def f_qc(obj_uni, obj, quant_tree, *args, **kwargs):
            '''gets quant child number {i} from quant tree of obj,
            sorting from i=0 as {age0} to i=-1 as {agef}.
            '''
            #print('f_qc called with uni, obj, args, kwargs:', obj_uni, obj, *args, **kwargs)
            __tracebackhide__ = self._tracebackhide
            child_tree  = quant_tree.get_child(i, oldest_first)
            if self.units_key is None:
                units_key = kwargs[UNITS_KEY_KWARG]
            else:
                units_key = self.units_key
            units_tuple = _units_lookup_by_quant_info(obj, child_tree.data, units_key)
            result      = units_tuple(obj_uni, obj, child_tree, *args, **kwargs)
            if return_type == 'units':
                return result.f
            elif return_type == 'name':
                return result.name
            elif return_type == 'tuple':
                return result
        # make pretty documentation for f_qc.
        youngest, oldest = 'youngest (added most-recently)', 'oldest (added first)'
        age0, agef       = (oldest, youngest) if oldest_first else (youngest, oldest)
        f_qc.__doc__  = f_qc.__doc__.format(i=i, age0=age0, agef=agef)
        f_qc.__name__ = 'child_' + str(i) + '__' + ('oldest' if oldest_first else 'youngest') + '_is_0'
        _special_args_info = 'arg[1] is assumed to be an obj with attribute got_vars_tree() which returns a QuantTree.'
        required_kwargs = [UNITS_KEY_KWARG] if self.units_key is None else []
        # return f_qc
        f_qc = self.FunclikeType(f_qc, argn=1, required_kwargs=required_kwargs, **self._kw__funclike_init)
        f_qc._special_args_info = lambda *args, **kwargs: _special_args_info
        return f_qc

    def quant_child(self, i, oldest_first=True):
        '''returns a Funclike which gets units tuple of i'th quant child in QUANTS_TREE for object=args[1].'''
        return self._quant_child(i, oldest_first, return_type='tuple')

    qc = quant_child  # alias

    def quant_child_units(self, i, oldest_first=True):
        '''returns a Funclike which gets units func for i'th quant child in QUANTS_TREE for object=args[1].'''
        return self._quant_child(i, oldest_first, return_type='units')

    qcu = quant_child_units  # alias

    def quant_child_name(self, i, oldest_first=True):
        '''returns a Funclike which gets units name for i'th quant child in QUANTS_TREE for object=args[1].'''
        return self._quant_child(i, oldest_first, return_type='name')

    qcn = quant_child_name  # alias


def _units_lookup_by_quant_info(obj, info, units_key=UNITS_UNIVERSAL_KEY,
                                default_f=_default_units_f, default_name=DEFAULT_UNITS_NAME):
    '''given obj, gets UnitsTuple from QuantInfo info.
    We are trying to get:
        obj.vardict[info.metaquant][info.typequant][info.quant][units_key].
    if we fail, try again with units_key = 'uni'.
        'uni' will be used to represent units which are the same in any system.

    if we fail again,
        return default     # which, by default, is the default UnitsTuple() defined in units.py.
    '''
    x = obj.quant_lookup(info)   # x is the entry in vardict for QuantInfo info.
    keys_to_check = (units_key, UNITS_UNIVERSAL_KEY)  # these are the keys we will try to find in x.
    # try to get units tuple:
    utuple = _multiple_lookup(x, *keys_to_check, default=None)
    if utuple is not None:
        return utuple
    # else:    # failed to get units tuple.
    # try to get units f and units name separately.
    keys_to_check_f    = [UNITS_KEY_F.format(key)    for key in keys_to_check]
    keys_to_check_name = [UNITS_KEY_NAME.format(key) for key in keys_to_check]
    msg_if_err = '\n  units_key = {}\n  quant_info = {}\n  quant_lookup_result = {}'.format(repr(units_key), info, x)
    u_f    = _multiple_lookup(x, *keys_to_check_f,    default=default_f(msg_if_err))
    u_name = _multiple_lookup(x, *keys_to_check_name, default=default_name)
    utuple_ = UnitsTuple(u_f, u_name)
    return utuple_

def _multiple_lookup(x, *keys, default=None):
    '''try to get keys from x. return result for first key found. return None if fail.'''
    for key in keys:
        result = x.get(key, None)
        if result is not None:
            return result
    return default

''' ----------------------------- Evaluate Units ----------------------------- '''

EvaluatedUnits = collections.namedtuple('EvaluatedUnits', ('factor', 'name'))

def get_units(obj, mode='si', **kw__units_f):
    '''evaluates units for most-recently-gotten var (at top of obj._quants_tree).
    Accepted modes are defined by UNITS_MODES in helita.sim.units.py near top of file.

    Accepted modes are 'si' for SI units, and 'cgs' for cgs units. Case-insensitive.
    '''
    # initial processing of mode.
    mode = mode.lower()
    assert mode in UNITS_MODES, 'Mode invalid! valid modes={}; got mode={}'.format(list(UNITS_MODES.keys()), repr(mode))
    units_key, format_attr = UNITS_MODES[mode]
    # lookup info.
    quant_info  = obj.get_quant_info(lookup_in_vardict=False)
    units_tuple = _units_lookup_by_quant_info(obj, quant_info, units_key=units_key)
    quant_tree  = obj.got_vars_tree(as_data=True)
    # evaluate units_tuple, given obj.uni, obj, and **kwargs.
    kw__units_f[ATTR_FORMAT_KWARG] = kw__units_f.get(ATTR_FORMAT_KWARG, format_attr)
    kw__units_f[UNITS_KEY_KWARG]   = kw__units_f.get(UNITS_KEY_KWARG,   units_key  )
    #print('in get_units, units_tuple is', units_tuple)
    result = units_tuple(obj.uni, obj, quant_tree, **kw__units_f)
    # make result formatting prettier and return result.
    result = EvaluatedUnits(result.f, str(result.name))
    return result



''' ----------------------------- Aliases ----------------------------- '''

# It can be helpful to import these aliases into other modules.
# for example, in a load_..._quantities file, you would do:
"""
from .units import (
    UNI, USI, UCGS, U_SYM, U_SYMS, U_TUPLE, DIMENSIONLESS, NO_UNITS
)
"""

# for making "universal" units
UNI     = UnitsFuncBuilder(units_key=None)  # , format_attr=None
# for making si units
USI     = UnitsFuncBuilder(units_key=UNITS_MODES['si'][0],  format_attr=UNITS_MODES['si'][1] )
# for making cgs units
UCGS    = UnitsFuncBuilder(units_key=UNITS_MODES['cgs'][0], format_attr=UNITS_MODES['cgs'][1])

# for making unit names ("UnitsExpression"s)
U_SYM   = UnitSymbol      # returns a single symbol
U_SYMS  = UnitSymbols     # returns multiple symbols

# for putting units info in vardict
U_TUPLE = UnitsTuple      # tuple with (units function, units expression)

# for dimensionless quantities
DIMENSIONLESS = DIMENSIONLESS_TUPLE  # alias
NO_UNITS      = DIMENSIONLESS_TUPLE  # alias