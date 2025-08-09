"""Architecture Main - 4-Layer Architecture Implementation

This module provides the main architecture orchestrator that manages
the 4-layer system and coordinates communication between layers.
"""

import asyncio
from typing import Optional, Dict, Any, List
from loguru import logger
from pathlib import Path

from .config_manager import ConfigManager
from .event_system import EventSystem, Event
from .adapter_manager import AdapterManager
from .layers import (
    BaseLayer, UserInterfaceLayer, IntelligenceHubLayer,
    AdapterLayer, ExecutionLayer, LayerFactory
)
from .types import Context, Intent, ExecutionPlan, ExecutionResult


class ClaudeVoiceAssistant:
    """
    Main architecture orchestrator for the Claude Voice Assistant.
    
    This class manages the 4-layer architecture and coordinates
    communication between all layers and components.
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 environment: Optional[str] = None):
        """
        Initialize the Claude Voice Assistant architecture.
        
        Args:
            config_dir: Configuration directory path
            environment: Environment name (dev, staging, prod)
        """
        # Core systems
        self.config_manager = ConfigManager(config_dir, environment)
        self.event_system = EventSystem()
        self.adapter_manager = AdapterManager(
            self.event_system,
            adapter_paths=["src/adapters"]  # Will be configured
        )
        
        # Architecture layers
        self.layers: Dict[str, BaseLayer] = {}
        
        # System state
        self.running = False
        self.initialized = False
        self._shutdown_event = asyncio.Event()
        
        # Current context
        self.current_context: Optional[Context] = None
    
    async def initialize(self) -> bool:
        """
        Initialize the complete architecture.
        
        Returns:
            True if initialization was successful
        """
        logger.info("Initializing Claude Voice Assistant Architecture")
        
        try:
            # Initialize core systems
            await self.config_manager.initialize()
            await self.event_system.initialize()
            await self.adapter_manager.initialize()
            
            # Create and initialize layers
            await self._initialize_layers()
            
            # Setup inter-layer communication
            await self._setup_layer_communication()
            
            # Setup event handlers
            await self._setup_event_handlers()
            
            self.initialized = True
            
            # Emit initialization event
            await self.event_system.emit(Event(
                event_type="architecture.initialized",
                data={
                    "layers": list(self.layers.keys()),
                    "adapters": len(self.adapter_manager._adapters)
                },
                source="architecture"
            ))
            
            logger.info("Architecture initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Architecture initialization failed: {e}")
            return False
    
    async def start(self) -> None:
        """Start the voice assistant system."""
        if not self.initialized:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize architecture")
        
        logger.info("Starting Claude Voice Assistant")
        self.running = True
        
        try:
            # Start the main processing loop
            await self._main_loop()
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown the voice assistant system."""
        logger.info("Shutting down Claude Voice Assistant")
        self.running = False
        
        # Signal shutdown
        self._shutdown_event.set()
        
        try:
            # Shutdown layers
            for layer_name, layer in self.layers.items():
                logger.debug(f"Shutting down layer: {layer_name}")
                await layer.shutdown()
            
            # Shutdown core systems
            await self.adapter_manager.shutdown()
            await self.event_system.shutdown()
            await self.config_manager.shutdown()
            
            # Emit shutdown event (if event system is still running)
            try:
                await self.event_system.emit(Event(
                    event_type="architecture.shutdown",
                    data={},
                    source="architecture"
                ))
            except:
                pass  # Event system might be already shutdown
            
            logger.info("Architecture shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def process_user_input(self, user_input: str) -> str:
        """
        Process user input through the 4-layer architecture.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Response message for the user
        """
        try:
            # Layer 1: UI Layer - Parse input to intent
            ui_layer = self.layers.get("user_interface")
            if not ui_layer:
                return "Error: User interface layer not available"
            
            intent = await ui_layer.process_input(user_input)
            
            # Layer 2: Intelligence Hub - Create execution plan
            hub_layer = self.layers.get("intelligence_hub")
            if not hub_layer:
                return "Error: Intelligence hub layer not available"
            
            # Update intent with AI understanding
            enhanced_intent = await hub_layer.understand_intent(intent)
            
            # Create execution plan
            execution_plan = await hub_layer.create_execution_plan(enhanced_intent)
            
            # Assess risks
            assessed_plan = await hub_layer.assess_risk(execution_plan)
            
            # Check if confirmation is needed
            if await hub_layer.should_require_confirmation(assessed_plan):
                confirmation = await ui_layer.get_user_confirmation(
                    f"This operation involves {len(assessed_plan.tasks)} tasks with risk level {assessed_plan.risk_level.value}. Continue?"
                )
                if not confirmation:
                    return "Operation cancelled by user"
            
            # Layer 3: Coordinate execution
            results = await hub_layer.coordinate_execution(assessed_plan)
            
            # Layer 4: Present results
            await ui_layer.present_results(results)
            
            # Learn from execution
            await hub_layer.learn_from_execution(assessed_plan, results)
            
            if results.overall_success:
                return f"Task completed successfully in {results.total_execution_time:.2f} seconds"
            else:
                return f"Task failed: {'; '.join(results.errors)}"
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return f"Error processing request: {str(e)}"
    
    async def _initialize_layers(self) -> None:
        """Initialize all architecture layers."""
        # Note: In a real implementation, these would be concrete implementations
        # For now, we're creating abstract base classes
        layer_types = [
            "user_interface",
            "intelligence_hub", 
            "adapter",
            "execution"
        ]
        
        for layer_type in layer_types:
            try:
                # Create layer instance (this would be concrete implementations)
                layer = LayerFactory.create_layer(layer_type, self.event_system)
                
                # Initialize the layer
                if await layer.initialize():
                    self.layers[layer_type] = layer
                    logger.info(f"Initialized layer: {layer_type}")
                else:
                    logger.error(f"Failed to initialize layer: {layer_type}")
                    
            except Exception as e:
                logger.error(f"Error initializing layer {layer_type}: {e}")
    
    async def _setup_layer_communication(self) -> None:
        """Setup communication channels between layers."""
        # Set up context sharing
        if self.current_context:
            for layer in self.layers.values():
                await layer.set_context(self.current_context)
    
    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for system events."""
        # Context update handler
        self.event_system.subscribe(
            "*.context_updated",
            self._handle_context_update,
            handler_id="architecture.context_handler"
        )
        
        # Layer message handler
        self.event_system.subscribe(
            "layer.message.*",
            self._handle_layer_message,
            handler_id="architecture.message_handler"
        )
        
        # Health monitoring
        self.event_system.subscribe(
            "*.health_check_failed",
            self._handle_health_check_failed,
            handler_id="architecture.health_handler"
        )
    
    async def _main_loop(self) -> None:
        """Main processing loop."""
        logger.info("Starting main processing loop")
        
        while self.running and not self._shutdown_event.is_set():
            try:
                # Update current context
                await self._update_system_context()
                
                # Health checks
                await self._perform_health_checks()
                
                # Wait for next iteration or shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=1.0
                    )
                    # Shutdown event was set
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    pass
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_system_context(self) -> None:
        """Update system context information."""
        # This would gather current system state, active applications, etc.
        # For now, create a basic context
        if not self.current_context:
            self.current_context = Context(
                user_id="default_user",
                session_id="current_session",
                environment_vars={}
            )
        
        # Update context in all layers
        for layer in self.layers.values():
            await layer.set_context(self.current_context)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        # Check layers
        for layer_name, layer in self.layers.items():
            try:
                if not await layer.health_check():
                    logger.warning(f"Layer {layer_name} failed health check")
            except Exception as e:
                logger.error(f"Health check error for layer {layer_name}: {e}")
    
    async def _handle_context_update(self, event: Event) -> None:
        """Handle context update events."""
        logger.debug(f"Context updated by {event.source}")
        
        # Propagate context updates to other layers if needed
        # This would implement more sophisticated context synchronization
    
    async def _handle_layer_message(self, event: Event) -> None:
        """Handle inter-layer messages."""
        message_data = event.data
        target_layer = message_data.get("target_layer")
        
        if target_layer in self.layers:
            logger.debug(f"Routing message to layer: {target_layer}")
            # Route message to target layer
            # This would implement message routing logic
    
    async def _handle_health_check_failed(self, event: Event) -> None:
        """Handle health check failures."""
        logger.warning(f"Health check failed: {event.data}")
        
        # Implement recovery logic
        # This could include restarting components, alerting users, etc.


# Convenience function for creating and running the assistant
async def run_claude_voice_assistant(config_dir: Optional[str] = None,
                                   environment: Optional[str] = None) -> None:
    """
    Run the Claude Voice Assistant.
    
    Args:
        config_dir: Configuration directory
        environment: Environment name
    """
    assistant = ClaudeVoiceAssistant(config_dir, environment)
    await assistant.start()