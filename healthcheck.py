
import os
import sys


def check():
    """Verifica se a aplicação está saudável"""
    try:
        # Verificar se arquivos necessários existem
        required_files = ['main.py', 'agriculture_dataset.csv']
        for file in required_files:
            if not os.path.exists(file):
                print(f"Arquivo não encontrado: {file}")
                return False

        # Verificar dependências
        import numpy
        import pandas

        print("Healthcheck: OK")
        return True

    except Exception as e:
        print(f"Healthcheck falhou: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if check() else 1)