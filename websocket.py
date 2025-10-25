#!/usr/bin/env python3
"""
Servidor WebSocket en Python para el sistema de traducción de lengua de señas
Conecta el frontend (pacientes y doctores) con la API Flask de predicción
"""

import asyncio
import websockets
import json
import logging
import requests
from typing import Dict, Set, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

# Configuración
WS_PORT = 8080
PREDICTION_API_URL = 'http://localhost:8000'
CHECK_API_INTERVAL = 30  # segundos

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ClientInfo:
    """Información de un cliente conectado"""
    ws: websockets.WebSocketServerProtocol
    session_id: str
    client_id: str
    client_type: str  # 'doctor' or 'patient'

@dataclass
class PredictionRequest:
    """Estructura para peticiones de predicción"""
    sequences: list
    threshold: float = 0.7

class WebSocketServer:
    def __init__(self):
        self.doctors: Dict[str, ClientInfo] = {}
        self.patients: Dict[str, ClientInfo] = {}
        self.sessions: Dict[str, Dict] = {}
        self.api_connected = False
        
    async def start_server(self):
        """Inicia el servidor WebSocket"""
        logger.info(f"🚀 Iniciando servidor WebSocket en puerto {WS_PORT}")
        logger.info(f"📡 API de predicción configurada en: {PREDICTION_API_URL}")
        
        # Iniciar verificación periódica de la API
        asyncio.create_task(self.periodic_api_check())
        
        # Iniciar servidor WebSocket
        async with websockets.serve(self.handle_client, "localhost", WS_PORT):
            logger.info("✅ Servidor WebSocket iniciado correctamente")
            await asyncio.Future()  # Mantener servidor activo
    
    async def handle_client(self, websocket):
        """Maneja una nueva conexión de cliente"""
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"🔗 Nueva conexión WebSocket desde {client_ip}")
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Cliente desconectado")
        except Exception as e:
            logger.error(f"❌ Error manejando cliente: {e}")
        finally:
            await self.cleanup_client(websocket)
    
    async def process_message(self, websocket, message):
        """Procesa un mensaje recibido del cliente"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            logger.info(f"📨 Mensaje recibido: {message_type}")
            
            if message_type == 'register':
                await self.handle_register(websocket, data)
            elif message_type == 'frame_data':
                await self.handle_frame_data(websocket, data)
            elif message_type == 'manual_translation':
                await self.handle_manual_translation(websocket, data)
            else:
                logger.warning(f"❓ Tipo de mensaje desconocido: {message_type}")
                await self.send_error(websocket, f"Tipo de mensaje desconocido: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decodificando JSON: {e}")
            await self.send_error(websocket, "Error decodificando mensaje JSON")
        except Exception as e:
            logger.error(f"❌ Error procesando mensaje: {e}")
            await self.send_error(websocket, "Error procesando mensaje")
    
    async def handle_register(self, websocket, data):
        """Maneja el registro de un nuevo cliente"""
        try:
            client_id = data.get('id')
            client_type = data.get('clientType')
            session_id = data.get('sessionId')
            
            if not client_id or not client_type:
                await self.send_error(websocket, "ID y clientType son requeridos")
                return
            
            client_info = ClientInfo(
                ws=websocket,
                session_id=session_id,
                client_id=client_id,
                client_type=client_type
            )
            
            # Registrar cliente según tipo
            if client_type == 'doctor':
                self.doctors[client_id] = client_info
                logger.info(f"👩‍⚕️ Doctor registrado: {client_id} en sesión {session_id}")
                
                # Enviar confirmación
                await websocket.send(json.dumps({
                    'type': 'registered',
                    'clientType': 'doctor',
                    'id': client_id,
                    'sessionId': session_id
                }))
                
                # Enviar estado actual de la API
                await websocket.send(json.dumps({
                    'type': 'api_status',
                    'connected': self.api_connected,
                    'timestamp': int(time.time() * 1000)
                }))
                
            elif client_type == 'patient':
                self.patients[client_id] = client_info
                logger.info(f"🏥 Paciente registrado: {client_id} en sesión {session_id}")
                
                # Enviar confirmación
                await websocket.send(json.dumps({
                    'type': 'registered',
                    'clientType': 'patient',
                    'id': client_id,
                    'sessionId': session_id
                }))
            
            # Actualizar sesiones
            if session_id:
                if session_id not in self.sessions:
                    self.sessions[session_id] = {'patient': None, 'doctor': None}
                self.sessions[session_id][client_type] = client_id
                
        except Exception as e:
            logger.error(f"❌ Error registrando cliente: {e}")
            await self.send_error(websocket, f"Error registrando cliente: {str(e)}")
    
    async def handle_frame_data(self, websocket, data):
        """Maneja datos de keypoints del paciente"""
        try:
            patient_id = data.get('patientId')
            session_id = data.get('sessionId')
            sequence = data.get('sequence', [])
            
            logger.info(f"📥 Keypoints recibidos del paciente {patient_id}: {len(sequence)} frames")
            
            if not sequence:
                logger.warning("⚠️ Secuencia vacía recibida")
                return
            
            # Enviar a API de predicción
            try:
                prediction = await self.get_prediction(sequence)
                
                if prediction and prediction.strip():
                    logger.info(f"🎯 Predicción obtenida: '{prediction}'")
                    
                    # Enviar predicción a todos los doctores de la sesión
                    await self.send_prediction_to_doctors(session_id, patient_id, prediction)
                else:
                    logger.info("⚪ Sin predicción válida para esta secuencia")
                    
            except Exception as prediction_error:
                logger.error(f"❌ Error procesando predicción: {prediction_error}")
                
                # Enviar error a doctores
                await self.send_prediction_error_to_doctors(session_id, str(prediction_error))
                
        except Exception as e:
            logger.error(f"❌ Error manejando frame_data: {e}")
    
    async def handle_manual_translation(self, websocket, data):
        """Maneja traducción manual enviada por el doctor"""
        try:
            session_id = data.get('sessionId')
            text = data.get('text', '')
            
            logger.info(f"📝 Traducción manual: {text}")
            
            # Reenviar a doctores de la sesión
            for doctor_id, doctor_info in self.doctors.items():
                if doctor_info.session_id == session_id:
                    await doctor_info.ws.send(json.dumps(data))
                    
        except Exception as e:
            logger.error(f"❌ Error manejando traducción manual: {e}")
    
    async def get_prediction(self, sequence):
        """Obtiene predicción de la API Flask"""
        try:
            # Preparar datos para la API
            request_data = {
                'sequences': [sequence],  # API espera array de secuencias
                'threshold': 0.7
            }
            
            logger.info(f"🔮 Enviando {len(sequence)} frames a API de predicción...")
            
            # Realizar petición HTTP síncrona (requests en hilo separado)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(
                    f"{PREDICTION_API_URL}/predict-batch",
                    json=request_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                # La API devuelve una lista directamente
                if isinstance(result, list) and len(result) > 0:
                    prediction = result[0]
                    logger.info(f"✅ Predicción recibida: '{prediction}'")
                    return prediction
                else:
                    logger.info("⚪ API no retornó predicciones")
                    return None
            elif response.status_code == 422:
                try:
                    error_detail = response.json().get('detail', 'Error de validación')
                except Exception:
                    error_detail = response.text
                logger.error(f"🔴 Error 422 en API: {error_detail}")
                return None
            else:
                logger.error(f"🔴 Error HTTP {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("⏰ Timeout conectando con API de predicción")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("🔌 Error de conexión con API de predicción")
            return None
        except Exception as e:
            logger.error(f"❌ Error inesperado en predicción: {e}")
            return None
    
    async def send_prediction_to_doctors(self, session_id, patient_id, prediction):
        """Envía predicción a todos los doctores de una sesión"""
        message = {
            'type': 'prediction_result',
            'sessionId': session_id,
            'patientId': patient_id,
            'prediction': prediction,
            'timestamp': int(time.time() * 1000),
            'confidence': 'high'
        }
        
        doctors_notified = 0
        for doctor_id, doctor_info in self.doctors.items():
            if doctor_info.session_id == session_id:
                try:
                    await doctor_info.ws.send(json.dumps(message))
                    doctors_notified += 1
                    logger.info(f"📤 Predicción enviada al doctor {doctor_id}")
                except Exception as e:
                    logger.error(f"❌ Error enviando predicción al doctor {doctor_id}: {e}")
        
        if doctors_notified == 0:
            logger.warning(f"⚠️ No se encontraron doctores para la sesión {session_id}")
    
    async def send_prediction_error_to_doctors(self, session_id, error_message):
        """Envía error de predicción a doctores"""
        message = {
            'type': 'prediction_error',
            'sessionId': session_id,
            'error': error_message,
            'timestamp': int(time.time() * 1000)
        }
        
        for doctor_id, doctor_info in self.doctors.items():
            if doctor_info.session_id == session_id:
                try:
                    await doctor_info.ws.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"❌ Error enviando error al doctor {doctor_id}: {e}")
    
    async def check_api_health(self):
        """Verifica el estado de la API Flask"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{PREDICTION_API_URL}/health", timeout=5)
            )
            
            if response.status_code == 200:
                logger.info("✅ API Flask conectada")
                return True
            else:
                logger.warning(f"⚠️ API Flask responde con código {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"❌ API Flask desconectada: {e}")
            return False
    
    async def periodic_api_check(self):
        """Verificación periódica del estado de la API"""
        while True:
            try:
                old_status = self.api_connected
                self.api_connected = await self.check_api_health()
                
                # Si cambió el estado, notificar a todos los doctores
                if old_status != self.api_connected:
                    await self.notify_api_status_to_doctors()
                
                await asyncio.sleep(CHECK_API_INTERVAL)
            except Exception as e:
                logger.error(f"❌ Error en verificación periódica de API: {e}")
                await asyncio.sleep(CHECK_API_INTERVAL)
    
    async def notify_api_status_to_doctors(self):
        """Notifica el estado de la API a todos los doctores"""
        message = {
            'type': 'api_status',
            'connected': self.api_connected,
            'timestamp': int(time.time() * 1000)
        }
        
        for doctor_id, doctor_info in self.doctors.items():
            try:
                await doctor_info.ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"❌ Error enviando estado API al doctor {doctor_id}: {e}")
    
    async def cleanup_client(self, websocket):
        """Limpia un cliente desconectado"""
        # Buscar y remover de doctors
        for doctor_id, doctor_info in list(self.doctors.items()):
            if doctor_info.ws == websocket:
                logger.info(f"🧹 Limpiando doctor: {doctor_id}")
                del self.doctors[doctor_id]
                break
        
        # Buscar y remover de patients
        for patient_id, patient_info in list(self.patients.items()):
            if patient_info.ws == websocket:
                logger.info(f"🧹 Limpiando paciente: {patient_id}")
                del self.patients[patient_id]
                break
    
    async def send_error(self, websocket, error_message):
        """Envía un mensaje de error al cliente"""
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': error_message,
                'timestamp': int(time.time() * 1000)
            }))
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje de error: {e}")

async def main():
    """Función principal"""
    server = WebSocketServer()
    await server.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")