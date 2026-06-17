# Reglas de Negocio — Simulador de Costes por Caso de Uso

## 1. Estructura General del Coste

```
Coste Total = CAPEX (único) + OPEX (mensual recurrente)
```

### CAPEX (Capital Expenditure) — desembolso único

| Componente | Origen |
|---|---|
| Integración de fuentes | Tabla Tipo × Complejidad |
| Capacidades del asistente | Coste fijo por capacidad |
| Cumplimiento (ENS) | Coste fijo por nivel |

### OPEX (Operational Expenditure) — coste mensual recurrente

| Componente | Origen |
|---|---|
| Mantenimiento de fuentes | 10% del coste de integración × multiplicador de frecuencia |
| Mantenimiento de capacidades | Coste fijo mensual por capacidad |
| Infraestructura cloud | Simulador AKS/API existente |

---

## 2. Costes de Integración de Fuentes

### 2.1. Tabla de costes por tipo y complejidad

| Tipo de fuente | Complejidad Baja | Complejidad Media | Complejidad Alta |
|---|---|---|---|
| **SharePoint / Alfresco** | 3.000 € | 5.000 € | 8.000 € |
| **Base de Datos (Oracle / SQL)** | 4.000 € | 6.500 € | 10.000 € |
| **Web Scraping** | 3.500 € | 5.500 € | 9.000 € |
| **API REST** | 2.500 € | 4.500 € | 7.000 € |
| **PDF Dinámico** | 2.000 € | 3.500 € | 6.000 € |

> Si el tipo de fuente o complejidad no está en la tabla, se aplica un valor por defecto de 5.000 €.

### 2.2. Complejidad — Criterios orientativos

| Complejidad | Criterios |
|---|---|
| **Baja** | Fuente con API estándar REST, documentación completa, formato estructurado. Integración directa sin transformación. |
| **Media** | Fuente con API propietaria, formato semiestructurado (JSON/XML), requiere transformación ligera o autenticación OAuth. |
| **Alta** | Fuente sin API pública (scraping complejo), datos no estructurados, requiere OCR/NLP, webs dinámicas con JavaScript, múltiples formatos de salida. |

---

## 3. Mantenimiento Mensual de Fuentes

```
Mantenimiento mensual = CosteIntegración × 10% × MultiplicadorFrecuencia
```

### 3.1. Multiplicadores por frecuencia de actualización

| Frecuencia | Multiplicador | Descripción |
|---|---|---|
| **Tiempo real** | 1.5 | Sincronización continua ante cualquier cambio. Mayor coste operativo. |
| **Cada hora** | 1.05 | Sincronización periódica cada hora. |
| **Diaria** | 1.02 | Sincronización una vez al día (referencia base). |
| **Semanal** | 0.5 | Sincronización semanal. Menor coste operativo. |
| **Mensual** | 0.2 | Sincronización mensual. Coste operativo mínimo. |

---

## 4. Costes de Capacidades del Asistente

| Capacidad | CAPEX (único) | OPEX (mensual) | Descripción |
|---|---|---|---|
| **IA Agéntica** (Tools / ejecución de tareas) | 8.500 € | 850 € | Capacidad del asistente para interactuar con backends, ejecutar tareas y usar herramientas externas. |
| **Anonimización en Tiempo Real** (Presidio) | 3.500 € | 350 € | Módulo de anonimización de datos personales en tiempo real para cumplimiento de protección de datos. |
| **Autenticación Corporativa** (SSO) | 4.000 € | 200 € | Integración con sistemas de identidad corporativos (SSO). |

---

## 5. Costes de Cumplimiento (ENS)

| Nivel ENS | CAPEX (único) | Descripción |
|---|---|---|
| **Ninguno** | 0 € | Sin certificación. |
| **Básico** | 1.000 € | Esquema Nacional de Seguridad — nivel básico. |
| **Medio** | 2.500 € | Esquema Nacional de Seguridad — nivel medio. |
| **Alto** | 5.000 € | Esquema Nacional de Seguridad — nivel alto. |

> Los costes ENS son únicamente CAPEX (no tienen componente mensual).

---

## 6. Coste de Infraestructura Cloud

El coste de infraestructura se obtiene del **simulador AKS/API existente** (`simulator.simulate_all`) utilizando los parámetros de negocio del caso de uso:

| Parámetro | Mapeo |
|---|---|
| `users` | Usuarios del caso de uso |
| `interactions_per_user_day` | Interacciones por usuario y día |
| `input_tokens_per_interaction` | Tokens de entrada por interacción |
| `output_tokens_per_interaction` | Tokens de salida por interacción |
| `working_days_per_month` | Días laborables al mes |
| `office_hours_per_day` | Horas de oficina al día |
| `peak_hours_per_day` | Horas pico al día |
| `concurrent_user_ratio` | Ratio de usuarios concurrentes |
| `peak_multiplier` | Multiplicador de pico |

### Opciones de despliegue

| Despliegue | Escenario simulado |
|---|---|
| **AKS Ideal (A100)** | `AKS UX Ideal` — GPU NVIDIA A100, máxima throughput |
| **AKS Económico (A10)** | `AKS UX Economico` — GPU NVIDIA A10, mejor coste/rendimiento |
| **API Azure OpenAI** | `API Azure OpenAI` — pago por uso, sin infraestructura |

> La simulación de infraestructura se ejecuta con `resize=True` y `mc_iterations=100` para obtener una estimación de coste mensual.

---

## 7. Fórmulas de Total

```
CAPE_total = IntegraciónFuentes + CapacidadesCAPEX + CumplimientoENS

OPEX_mensual = MantenimientoFuentes + CapacidadesOPEX + Infraestructura

Año 1 = CAPEX_total + OPEX_mensual × 12

Año 3 = CAPEX_total + OPEX_mensual × 36
```

---

## 8. Notas Técnicas

- **Arquitectura asumida**: RAG Híbrida (vectorial + grafo) con sincronización automática.
- **No se contempla ingesta estática** (carga manual de PDFs o archivos locales). Todas las fuentes se conectan en tiempo real o mediante sincronización automática.
- **Mantenimiento incluido**: Los costes incluyen mantenimiento evolutivo y correctivo durante el periodo de garantía.
- **Moneda**: Todos los costes están expresados en euros (€).
- **Actualización de precios**: Los precios de infraestructura Azure se obtienen de la API Retail Prices de Azure en tiempo real (salvo que se proporcionen valores manuales).

---

*Documento generado para el Simulador de Costes Azure AKS + LLM v2.0.0*
