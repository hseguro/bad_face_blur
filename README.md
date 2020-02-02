# Instalar 'faced'
`pip install git+https://github.com/hseguro/faced.git`

# Instalar dependencias
`pip install -r requirements.txt`

# Iniciar procesado
`DEBUG=1` para ver una previsualización del video

`DEBUG=0` para solo procesar

`BLUR=3` Para difuminar la imagen (numero impar)

`FACED_ACC=0.6` Accuracy

`DEBUG=1 python process.py input.mp4 output.mp4`