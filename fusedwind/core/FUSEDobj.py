import json

def flatten(v):
    """Flatten a nested dictionary

    Parameters
    ----------
    v:  dict
        The structure to flatten

    Returns
    -------
    d:  dict
        The nested dictionary

    Example
    -------
    >> flatten({'layout': [
              {
                  'name': 'wt1',
                  'hub_height': 80.,
                  'rotor_diameter': 80.,
                  'position': {'x':6400, 'y':23232},
                  'power': 2000.,
              },
              {
                  'name': 'wt2',
                  'hub_height': 70.,
                  'rotor_diameter': 80.,
                  'position': {'x':6400, 'y':23332},
                  'power': 2000.,
              },
          ]
      })
    >>
    {'layout:wt1:hub_height': 80.0,
     'layout:wt1:name': 'wt1',
     'layout:wt1:position:x': 6400,
     'layout:wt1:position:y': 23232,
     'layout:wt1:power': 2000.0,
     'layout:wt1:rotor_diameter': 80.0,
     'layout:wt2:hub_height': 70.0,
     'layout:wt2:name': 'wt2',
     'layout:wt2:position:x': 6400,
     'layout:wt2:position:y': 23332,
     'layout:wt2:power': 2000.0,
     'layout:wt2:rotor_diameter': 80.0}

    """

    def _flatten(v, k=''):
        """Recursive generator used to flatten a nested dictionary

        Parameters
        ----------
        v:  [dict|list|int|float]
            The structure to flatten
        k:  The base key to flatten it with
        """
        if k == '':
            base = ''
        else:
            base = k+':'

        if isinstance(v, dict):
            for k2, v2 in v.items():
                for k3, v3 in _flatten(v2, base+k2):
                    yield (k3, v3)
        elif isinstance(v, list):
            if isinstance(v[0], dict):
                if 'name' in v[0]:
                    try:
                        for i in v:
                            for k2, v2 in _flatten(i, base + i['name']):
                                yield (k2, v2)
                    except Exception as e:
                        print('flattening failed:',k, v, str(e))
                        yield (k, v)
                else:
                    yield (k, v)
            else:
                yield (k, v)
        else:
            yield (k, v)

    return dict(_flatten(v))


class FUSEDobj(object):
    """A class to handle nested dictionaries in an openMDAO friendly way.
    """
    def __init__(self, val, name=None, parent=None, root=None):
        """
        Parameters
        ----------
        val: [dict|list|int|float]
             The nested structure to represent
        name: str
            The name of the structure
        parent: FUSEDobj
            The parent object in the original nested structure
        root: FUSEDobj
            Shortcut to the root of the structure
        """
        if not root:
            root = self
            self._root = None
            self._flat_dict = flatten(val)
        else:
            self._root = root
        self._type = val.__class__.__name__
        self._parent = parent
        self._name = name
        self._keys = []
        if isinstance(val, dict):
            for k, v in val.items():
                setattr(self, k, FUSEDobj(v, k, self, root))
                self._keys.append(k)
        elif isinstance(val, list):
            if isinstance(val[0], dict):
                if 'name' in val[0]:
                    try:
                        for v in val:
                            setattr(self, v['name'], FUSEDobj(v, v['name'], self, root))
                            self._keys.append(v['name'])
                        self._type = 'list(dict)'
                    except Exception as e:
                        print(val,'is confusing, looks like a list of items, but this happened:', str(e))
                        self._type = 'list'
                else:
                    self._type = 'list'
            else:
                self._type = 'list'


    def __repr__(self):
        """Pretty print in jupyter notebook"""
        if self._type == 'list(dict)' or self._type == 'dict':
            return self.json()
        else:
            return str(self.value)

    def json(self):
        """Return a json representation of the object"""
        return json.dumps(self._json_nest(), indent=2)

    def _json_nest(self):
        """Re construct the nested structure"""
        if self._type == 'list(dict)':
            return [i._json_nest() for i in self.values()]
        if self._type == 'dict':
            return {k: v._json_nest() for k, v in self.items()}
        # default return is the object value
        if self._type == 'ndarray':
            return self.value.tolist()
        else:
            return self.value

    def nest(self):
        """Re construct the nested structure"""
        if self._type == 'list(dict)':
            return [i.nest() for i in self.values()]
        if self._type == 'dict':
            return {k: v.nest() for k, v in self.items()}
        # default return is the object value
        return self.value

    @property
    def value(self):
        """Get the value from the root flatten dictionary"""
        if self.address in self._root._flat_dict:
            return self._root._flat_dict[self.address]
        else:
            raise Exception('This is a nested object, use nest() instead')



    @value.setter
    def value(self, val):
        """Write the value at the write place"""
        if self._root:
            self._root._flat_dict[self.address] = val
        else:
            if self.address in self._flat_dict:
                self._flat_dict[self.address] = val
            else:
                raise Exception('This is a nested object, use nest() instead')


    def __setattr__(self, key, val):
        if hasattr(self, key):
            obj = getattr(self, key)
            if isinstance(obj, FUSEDobj):
                if obj._type == 'dict':
                    if isinstance(val, dict):
                        for k, v in val.items():
                            setattr(obj, k, v)
                    else:
                        raise Exception(v,'should be a dict but instead is a', v.__class__.__name__)
                elif obj._type == 'list(dict)':
                    if isinstance(val, list):
                        for i in val:
                            if isinstance(i, dict):
                                if 'name' in i:
                                    setattr(obj, i['name'], i)
                                else:
                                    raise Exception(i,'has no name')
                            else:
                                raise Exception(val[i], 'should be a dict')
                else:
                    # write in the value
                    obj.value = val
            else: # Normal behavior
                super(FUSEDobj, self).__setattr__(key, val)
        else: # Normal behavior
            super(FUSEDobj, self).__setattr__(key, val)

    @property
    def address(self):
        if self._name == None:
            return ''
        if self._parent == None:
            return self._name
        if self._parent.address == '':
            return self._name
        return ':'.join([self._parent.address, self._name])

    def resolves(self, val):
#        print val
        address = val.split(':*:')
        if len(address) == 1:
            keys = val.split(':')
            #print keys
            if keys[0] == '':
                yield
            elif len(keys) == 1:
                if keys[0] == '*':
                    for k in self.values():
                        yield k
                else:
                    yield getattr(self, keys[0])
            else:
                if keys[0] == '*':
                    for v in self.values():
                        for i in v.resolves(':'.join(keys[1:])):
                            if isinstance(i, FUSEDobj):
                                yield i
                            else:
                                for j in i:
                                    yield j
                else:
                    for i in getattr(self, keys[0]).resolves(':'.join(keys[1:])):
                        if isinstance(i, FUSEDobj):
                            yield i
                        else:
                            for j in i:
                                yield j
        else:
            for i in self.resolves(address[0]):
                for k in i.values():
                    for l in k.resolves(':'.join(address[1:])):
                        yield l

    def keys(self):
        if hasattr(self, '_keys'):
            for k in self._keys:
                yield k
        elif self._type == 'list(dict)':
            for i in self._keys:
                yield getattr(self, i)._name
        else:
            raise Exception('No keys are available')

    def values(self):
        for k in self.keys():
            yield getattr(self, k)

    def items(self):
        for k in self.keys():
            yield k, getattr(self, k)

    ## Operators overloads -----------------------------------------------------


    def __add__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__add__(other.value)
            else:
                return self.value.__add__(other)
        except:
            return super(FUSEDobj, self).__add__(other)

    def __sub__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__sub__(other.value)
            else:
                return self.value.__sub__(other)
        except:
            return super(FUSEDobj, self).__sub__(other)

    def __mult__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__mult__(other.value)
            else:
                return self.value.__mult__(other)
        except:
            return super(FUSEDobj, self).__mult__(other)

    def __mul__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__mul__(other.value)
            else:
                return self.value.__mul__(other)
        except:
            return super(FUSEDobj, self).__mul__(other)

    def __matmul__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__matmul__(other.value)
            else:
                return self.value.__matmul__(other)
        except:
            return super(FUSEDobj, self).__matmul__(other)

    def __truediv__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__truediv__(other.value)
            else:
                return self.value.__truediv__(other)
        except:
            return super(FUSEDobj, self).__truediv__(other)

    def __floordiv__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__floordiv__(other.value)
            else:
                return self.value.__floordiv__(other)
        except:
            return super(FUSEDobj, self).__floordiv__(other)

    def __mod__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__mod__(other.value)
            else:
                return self.value.__mod__(other)
        except:
            return super(FUSEDobj, self).__mod__(other)

    def __divmod__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__divmod__(other.value)
            else:
                return self.value.__divmod__(other)
        except:
            return super(FUSEDobj, self).__divmod__(other)

    def __pow__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__pow__(other.value)
            else:
                return self.value.__pow__(other)
        except:
            return super(FUSEDobj, self).__pow__(other)

    def __lshift__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__lshift__(other.value)
            else:
                return self.value.__lshift__(other)
        except:
            return super(FUSEDobj, self).__lshift__(other)

    def __rshift__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__rshift__(other.value)
            else:
                return self.value.__rshift__(other)
        except:
            return super(FUSEDobj, self).__rshift__(other)

    def __and__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__and__(other.value)
            else:
                return self.value.__and__(other)
        except:
            return super(FUSEDobj, self).__and__(other)

    def __xor__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__xor__(other.value)
            else:
                return self.value.__xor__(other)
        except:
            return super(FUSEDobj, self).__xor__(other)

    def __or__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__or__(other.value)
            else:
                return self.value.__or__(other)
        except:
            return super(FUSEDobj, self).__or__(other)

    def __lt__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__lt__(other.value)
            else:
                return self.value.__lt__(other)
        except:
            return super(FUSEDobj, self).__lt__(other)

    def __le__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__le__(other.value)
            else:
                return self.value.__le__(other)
        except:
            return super(FUSEDobj, self).__le__(other)

    def __eq__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__eq__(other.value)
            else:
                return self.value.__eq__(other)
        except:
            return super(FUSEDobj, self).__eq__(other)

    def __ne__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__ne__(other.value)
            else:
                return self.value.__ne__(other)
        except:
            return super(FUSEDobj, self).__ne__(other)

    def __gt__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__gt__(other.value)
            else:
                return self.value.__gt__(other)
        except:
            return super(FUSEDobj, self).__gt__(other)

    def __ge__(self, other):
        try:
            if isinstance(other, FUSEDobj):
                return self.value.__ge__(other.value)
            else:
                return self.value.__ge__(other)
        except:
            return super(FUSEDobj, self).__ge__(other)

    def __neg__(self):
        try:
            return self.isinstance.__ge__()
        except:
            return super(FUSEDobj, self).__neg__()

    def __pos__(self):
        try:
            return self.value.__pos__()
        except:
            return super(FUSEDobj, self).__pos__()

    def __abs__(self):
        try:
            return self.value.__abs__()
        except:
            return super(FUSEDobj, self).__abs__()

    def __invert__(self):
        try:
            return self.value.__invert__()
        except:
            return super(FUSEDobj, self).__invert__()

    def __complex__(self):
        try:
            return self.value.__complex__()
        except:
            return super(FUSEDobj, self).__complex__()

    def __int__(self):
        try:
            return self.value.__int__()
        except:
            return super(FUSEDobj, self).__int__()

    def __float__(self):
        try:
            return self.__float__()
        except:
            return super(FUSEDobj, self).__float__()

    def __round__(self):
        try:
            return self.value.__round__()
        except:
            return super(FUSEDobj, self).__round__()
