from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np
import math

class Celda(Agent):
    def __init__(self, unique_id, model, suciedad: bool = False):
        super().__init__(unique_id, model)
        self.sucia = suciedad


class Cargador(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Mueble(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class RobotLimpieza(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sig_pos = None
        self.movimientos = 0
        self.recargas = 0
        self.carga = 100

    def limpiar_una_celda(self, lista_de_celdas_sucias):
        celda_a_limpiar = self.random.choice(lista_de_celdas_sucias)
        celda_a_limpiar.sucia = False
        self.sig_pos = celda_a_limpiar.pos

    def seleccionar_nueva_pos(self, lista_de_vecinos):
        posiciones_ocupadas = [robot.sig_pos for robot in self.model.schedule.agents if isinstance(robot, RobotLimpieza)]
        posiciones_libres = [vecino for vecino in lista_de_vecinos if vecino.pos not in posiciones_ocupadas]

        if posiciones_libres:
            nueva_pos = self.random.choice(posiciones_libres).pos
            # Evitar colisiones marcando la posición elegida
            self.sig_pos = nueva_pos
        else:
            # Si no hay posiciones libres, permanecer en la posición actual
            self.sig_pos = self.pos

    @staticmethod
    def buscar_celdas_sucia(lista_de_vecinos):
        celdas_sucias = list()
        for vecino in lista_de_vecinos:
            if isinstance(vecino, Celda) and vecino.sucia:
                celdas_sucias.append(vecino)
        return celdas_sucias
    
    def calcular_distancia(self, cargador, elemento_actual):
        # Utiliza la fórmula de distancia euclidiana para calcular la distancia entre dos puntos
        x1, y1 = cargador
        x2 = elemento_actual[0]
        y2 = elemento_actual[1]
        distancia = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distancia

    def buscar_cercana(self,pos_cargador,posiciones):

        distancia_minima = float('inf')
        pos_mas_cercano = None

        for posicion in posiciones:
            distancia = self.calcular_distancia(posicion, pos_cargador)
            if distancia < distancia_minima:
                distancia_minima = distancia
                pos_mas_cercano = posicion
        return pos_mas_cercano
    
    def buscar_cargadores(self,elemento_actual):
        if not self.model.posiciones_cargadores:
            return None  # Si la lista de posiciones de cargadores está vacía, no hay cargadores disponibles

        distancia_minima = float('inf')
        cargador_mas_cercano = None

        for cargador in self.model.posiciones_cargadores:
            distancia = self.calcular_distancia(cargador, elemento_actual)
            if distancia < distancia_minima:
                distancia_minima = distancia
                cargador_mas_cercano = cargador
        return cargador_mas_cercano
    
    def moverse_a_cargador(self, pos_cargador, celdas_sucias, vecinos):

        pos_vecinos=[]
        vecinos2 = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        for vecino in vecinos2:
            pos_vecinos.append(vecino.pos)
        self.sig_pos=self.buscar_cercana(pos_cargador,pos_vecinos)

        #si hay obstaculos
        posiciones_ocupadas = [robot.sig_pos for robot in self.model.schedule.agents if isinstance(robot, RobotLimpieza) and robot != self]
        robots = [robot for robot in self.model.schedule.agents if isinstance(robot, RobotLimpieza)]
        posiciones_libres = [vecino.pos for vecino in vecinos if vecino.pos not in posiciones_ocupadas]

        if self.sig_pos in self.model.posiciones_cargadores and self.sig_pos in posiciones_ocupadas:
            for robot in robots:
                if robot.sig_pos==self.sig_pos:
                   
                    if robot.carga<self.carga or robot.pos==pos_cargador:
                        self.sig_pos=self.pos
                    elif robot.carga>= self.carga:
                        robot.sig_pos=robot.pos
        elif self.sig_pos not in self.model.posiciones_cargadores :
            if posiciones_libres:
                if self.sig_pos not in posiciones_libres:
                    if self.sig_pos not in self.model.posiciones_cargadores:
                        self.sig_pos=self.buscar_cercana(pos_cargador,posiciones_libres)
            else:
                self.sig_pos=self.pos

        #si esta sucia    
        for celda in celdas_sucias:
            if self.sig_pos == celda.pos:
                celda.sucia=False

    def cargar(self):
        self.carga+=25
        if self.carga>=99:
            self.carga=99
        self.sig_pos=self.pos

    def step(self):
        vecinos = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)

        for vecino in vecinos:
            if isinstance(vecino, Mueble):
                vecinos.remove(vecino)
        
        for vecino in vecinos:
            if vecino.pos in self.model.posiciones_cargadores:
                vecinos.remove(vecino)
        

        celdas_sucias = self.buscar_celdas_sucia(vecinos)

        if self.pos in self.model.posiciones_cargadores:#esta en un cargador
            if self.carga<99: #cargar
                self.cargar()  
            else: #ya termino de cargar
                self.recargas+=1
                if len(celdas_sucias) == 0:
                    self.seleccionar_nueva_pos(vecinos)
                else:
                    self.limpiar_una_celda(celdas_sucias)
        else:#no esta en un cargador
            if self.carga<60:
                pos_cargador=self.buscar_cargadores(self.pos)
                self.moverse_a_cargador(pos_cargador, celdas_sucias, vecinos)
            else:
                if len(celdas_sucias) == 0:
                    self.seleccionar_nueva_pos(vecinos)
                else:
                    self.limpiar_una_celda(celdas_sucias)

    def advance(self):
        if self.pos != self.sig_pos:
            self.movimientos += 1

        if self.carga > 0:
           
            self.model.grid.move_agent(self, self.sig_pos)
            
            if self.pos not in self.model.posiciones_cargadores:
                self.carga -= 1
            
    
class Habitacion(Model):
    def __init__(self, M: int, N: int,
                 num_agentes: int = 5,
                 porc_celdas_sucias: float = 0.6,
                 porc_muebles: float = 0.1,
                 modo_pos_inicial: str = 'Fija',
                 ):

        self.num_agentes = num_agentes
        self.porc_celdas_sucias = porc_celdas_sucias
        self.porc_muebles = porc_muebles
        
        self.grid = MultiGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)


    # posicionamiento de cargadores
        self.posiciones_cargadores= ((0, 0), (0, N-1), (M-1, 0), (M-1, N-1))

        for id, pos in enumerate(self.posiciones_cargadores):
            cargador = Cargador(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(cargador, pos)

        global posiciones_disponibles
        posiciones_disponibles = [pos for _, pos in self.grid.coord_iter() if pos not in self.posiciones_cargadores]

        # Posicionamiento de muebles
        num_muebles = int(M * N * porc_muebles)
        posiciones_muebles = self.random.sample(posiciones_disponibles, k=num_muebles)
        for id, pos in enumerate(posiciones_muebles):
            mueble = Mueble(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)


        # Posicionamiento de celdas sucias
        self.num_celdas_sucias = int(M * N * porc_celdas_sucias)
        posiciones_celdas_sucias = self.random.sample(
            posiciones_disponibles, k=self.num_celdas_sucias)

        for id, pos in enumerate(posiciones_disponibles):
            suciedad = pos in posiciones_celdas_sucias
            celda = Celda(int(f"{num_agentes}{id}") + 1, self, suciedad)
            self.grid.place_agent(celda, pos)

        # Posicionamiento de agentes robot
        if modo_pos_inicial == 'Aleatoria':
            pos_inicial_robots = self.random.sample(posiciones_disponibles, k=num_agentes)
        else:  # 'Fija'
            pos_inicial_robots = [(10, 3)] * num_agentes

        for id in range(num_agentes):
            robot = RobotLimpieza(id, self)
            self.grid.place_agent(robot, pos_inicial_robots[id])
            self.schedule.add(robot)


        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid, "Cargas": get_cargas,
                             "CeldasSucias": get_sucias},
        )

    def step(self):
        self.datacollector.collect(self)

        self.schedule.step()

        if self.todoLimpio():
            print(f"At step {self.schedule.steps}: All cells are clean!")

    # Imprimir los movimientos de cada agente
            for robot in self.schedule.agents :
                if isinstance(robot, RobotLimpieza):
                    dic=get_movimientos(robot)                   
                    dic2=get_recargas(robot)

                    for agente_id, movimientos in dic.items():
                        print(f"Agente {agente_id}: {movimientos} movimientos.")
                    for agente_id, recargas in dic2.items():
                        print(f"Agente {agente_id}: {recargas} recargas.")
            self.running = False  # Detener la simulación


    def todoLimpio(self):
        for (content, x) in self.grid.coord_iter():
            for obj in content:
                if isinstance(obj, Celda) and obj.sucia:
                    return False
        return True

def get_grid(model: Model) -> np.ndarray:
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        x, y = pos
        for obj in cell_content:
            if isinstance(obj, RobotLimpieza):
                grid[x][y] = 2
            elif isinstance(obj, Celda):
                grid[x][y] = int(obj.sucia)
    return grid


def get_cargas(model: Model):
    return [(agent.unique_id, agent.carga) for agent in model.schedule.agents]


def get_sucias(model: Model) -> int:
    sum_sucias = 0
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        for obj in cell_content:
            if isinstance(obj, Celda) and obj.sucia:
                sum_sucias += 1
    return sum_sucias / model.num_celdas_sucias


def get_movimientos(agent: Agent) -> dict:
    if isinstance(agent, RobotLimpieza):
        return {agent.unique_id: agent.movimientos}
    
def get_recargas(agent: Agent) -> dict:
    if isinstance(agent, RobotLimpieza):
        return {agent.unique_id: agent.recargas}


