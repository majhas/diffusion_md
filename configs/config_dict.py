from types import SimpleNamespace


class ConfigDict(SimpleNamespace):
    def __init__(self, **kwargs):
        super(ConfigDict, self).__init__(**kwargs)

        pass

    def get(self, name, default=None):
        try:
            return self.name
        except:
            return default


if __name__ == '__main__':
    config_dict = ConfigDict(**{'prop': 'cool'})
    
    print(config_dict.prop)
    print(config_dict.get('prop'))
    print(config_dict.get('something', 'I do not have it'))