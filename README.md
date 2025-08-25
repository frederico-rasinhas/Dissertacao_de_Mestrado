# Localização e Navegação em Ambiente GNSS-denied com LIDAR e Odometria Visual

Neste repositório é apresentado o código desenvolvido no âmbito da dissertação de mestrado em Ciências Militares Navais, na especialidade de Engenharia Naval – Ramo de Armas e Eletrónica. O trabalho centra-se na implementação de um sistema de localização e navegação em cenários com disrupção de sinal GNSS, integrando dados provenientes de encoders presentes nas rodas do veiculo, uma IMU, odometria visual e LIDAR através de um Error-State Extended Kalman Filter.

<p align="center">
  <img src="![Image](https://github.com/user-attachments/assets/0a7f5594-032d-4fcd-96eb-621b0416df42)" width="300" height="300">
</p>

## Descrição
 
O objetivo deste trabalho consistiu no desenvolvimento de uma arquitetura robusta de localização e navegação para veículos terrestres não tripulados (UGVs) em ambientes caracterizados por disrupção ou ausência de sinal GNSS. Para tal, foram implementados algoritmos de odometria visual monocular e de odometria LIDAR, posteriormente integrados com a informação proveniente da IMU, dos encoders das rodas do veículo e, de forma ocasional ou mesmo inexistente, do sinal GNSS, através de um Error-State Extended Kalman Filter (ESKF), seguindo a formulação de Joan Solà (2017) e adaptada a partir de [bdamas/ESKF](https://github.com/bdamas/ESKF). 


## Código
A implementação foi realizada em Python e está organizada da seguinte forma:

- **odometria_visual** – Algoritmos de estimação de pose com base em visão computacional monocular.  
    - **VO.py** – Implementa um sistema de odometria visual monocular em pós-processamento, lendo imagens comprimidas de um rosbag, extraindo e rastreando features entre frames consecutivos, estimando incrementos de pose com base na matriz essencial e numa restrição de plano para recuperação de escala.

- **odometria_lidar** – Algoritmos de estimação de pose utilizando dados LIDAR.  
    - **Lidar.py** – Reconstrução de trajetória 2D com LiDAR, aplicando voxel downsampling, extração de arestas e planos e ICP robusto.

- **ESKF_core** – Núcleo da implementação do ESKF, incluindo as funções de observação e correção de cada sensor.  
    - **dataloader.py** – Módulo de carregamento unificado dos dados provenientes dos diversos sensores, organizados cronologicamente.  
    - **eskf.py** – Implementação central do ESKF, com definição de classes abstratas de modelos dinâmicos e sensores.  
    - **imuDynamics.py** – Modelo dinâmico da IMU, incluindo o estados nominal, o estado de erro e respetivas equações de previsão.  
    - **GnssSensor.py** – Modelo de sensor GNSS, com conversão de coordenadas geodésicas para ENU e transformação para o referencial do rover.  
    - **LIDARSensor.py** – Modelo de sensor LIDAR, com conversão de incrementos em velocidades no corpo, atualização de orientação e ajuste dinâmico da covariância com base no número de caracteristicas detetadas.  
    - **OdometrySensor.py** – Modelo de sensor de odometria de rodas, convertendo velocidades linear e angular em observações no corpo e orientação em quaterniões.  
    - **VOSensor.py** – Modelo de sensor de odometria visual, convertendo incrementos em velocidades no corpo, atualização de orientação em quaterniões e ajuste da covariância com base no número de correspondências entre frames.

- **ESKF_main** – Scripts principais de execução do ESKF, responsáveis pelos ciclos completos de predição e correção, integrando a IMU e os diferentes sensores (VO, LIDAR, odometria de rodas e GNSS) nas suas múltiplas combinações.

Os ficheiros rosbag utilizados nos cenários experimentais encontram-se disponíveis em:  
https://drive.google.com/drive/folders/16EBPGkYK5u9Sbr1qC-6h7XCL1-PKYs50?usp=sharing

## Autor
Frederico Rasinhas  

email:  
- fredericoalvesrasinhas@gmail.com


