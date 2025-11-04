import importlib, traceback

modules = ['function_app', 'function_app_obras', 'app', 'app_obras']
for name in modules:
    try:
        print('\n--- tentando importar', name)
        m = importlib.import_module(name)
        print(name, 'importado com sucesso. funções no módulo (primeiras 30):')
        items = [a for a in dir(m) if callable(getattr(m, a)) or 'function' in a.lower() or 'search' in a.lower()][:30]
        print(items)
    except Exception:
        print('ERRO importando', name)
        traceback.print_exc()
