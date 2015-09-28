__author__ = 'pire'


def flatten(k, v):
    if v['type'] == 'vartree':
        for k2, v2 in v['val'].items():
            for k3, v3 in flatten(k+':'+k2, v2):
                yield (k3, v3)
    else:
        yield (k, v['val'])

class FlatTree(object):
    def __init__(self, dic, name=''):
        self._dic = dic
        self._name = name

    def _extend_name(self, item):
        if self._name == '':
            return item
        else:
            return self._name + ':' + item

    def contains(self, name):
        return any(any(name == k for k in s.split(':')) for s in self.keys())

    def __getitem__(self, item):
        name = self._extend_name(item)

        if name in self.keys():
            return self._dic[name]

        if self.contains(name):
            return FlatTree(self, name)

        return self._dic[name]


    def __setitem__(self, key, value):
        self._dic[self._extend_name(key)] = value

    def keys(self):
        return self._dic.keys()

    def items(self):
        return self._dic.items()

    def iteritems(self):
        for k,v in self._dic.iteritems():
            yield k,v

    def __str__(self):
        return self._dic.__str__()

    def __getattr__(self, item):
        """

        """
        if item[0] == '_':
            return super(FlatTree, self).__getattribute__(item)

        return self.__getitem__(item)

        #TODO: fix potentiel bug if there is a key that has the same name as a vartree root.
        ## f.ex {kat':'leopold', 'kat:kat':'bernard'}

        #TODO: find a way to raise an error if an end-key is not in the dic

    def __setattr__(self, item, value):
        if item[0] == '_':
            return super(FlatTree, self).__setattr__(item, value)
        self.__setitem__(item, value)



class branch(object):
    def __init__(self, dic, flat_dic={}, name=''):
        self._name = name
        self._dic = dic
        self._flat_dic = flat_dic
        if len(flat_dic) == 0:
            self._create_flat_dic()

        for k, v in dic.items():
            if v['type'] == 'vartree':
                super(branch, self).__setattr__(k, branch(v['val'], self._flat_dic, self._get_name(k)))

    def _create_flat_dic(self):
        for k, v in self._dic.items():
            for k2, v2 in flatten(self._get_name(k),v):
                self._flat_dic[k2] = v2

    def _get_name(self, name):
        if self._name == '':
            return name
        else:
            return self._name + ':' + name

    def __getattr__(self, item):
        if item.startswith('_'):
            self.__getattribute__(item)
        else:
            try:
                self._flat_dic[self._get_name(item)]
            except:
                self.__getattribute__(item)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super(branch, self).__setattr__(key, value)
        else:
            name = self._get_name(key)
            if name in self._flat_dic:
                self._flat_dic[name] = value
            else: #should this fail?
                raise KeyError(key+' is not permitted')
                #super(branch, self).__setattr__(key, value)
