"""
Nexlify Enhanced - Uniswap V3 Integration
Implements Feature 17: DEX integration for decentralized trading
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime
import json
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
import aiohttp

logger = logging.getLogger(__name__)

# Uniswap V3 Contract Addresses (Ethereum Mainnet)
UNISWAP_V3_ADDRESSES = {
    'factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
    'router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
    'quoter': '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
    'position_manager': '0xC36442b4a4522E871399CD717aBDD847Ab11FE88'
}

# Common Token Addresses
TOKENS = {
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
    'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'
}

class UniswapV3Integration:
    """
    Uniswap V3 integration for decentralized trading
    Supports swaps, liquidity provision, and arbitrage detection
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Uniswap integration
        
        Args:
            config: Configuration with RPC URL, private key, etc.
        """
        self.config = config
        self.w3 = self._setup_web3()
        self.account = self._setup_account()
        
        # Load contract ABIs
        self.contracts = self._load_contracts()
        
        # Gas price strategy
        self.gas_price_multiplier = config.get('gas_price_multiplier', 1.1)
        self.max_gas_price = config.get('max_gas_price', 300)  # Gwei
        
        # Trading parameters
        self.slippage_tolerance = config.get('slippage_tolerance', 0.005)  # 0.5%
        self.deadline_seconds = config.get('deadline_seconds', 300)  # 5 minutes
        
        # Pool cache
        self.pool_cache = {}
        self.last_pool_update = {}
        
    def _setup_web3(self) -> Web3:
        """Setup Web3 connection"""
        rpc_url = self.config.get('rpc_url', 'https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY')
        
        if rpc_url.startswith('http'):
            w3 = Web3(Web3.HTTPProvider(rpc_url))
        elif rpc_url.startswith('ws'):
            w3 = Web3(Web3.WebsocketProvider(rpc_url))
        else:
            raise ValueError(f"Invalid RPC URL: {rpc_url}")
            
        # Add middleware for PoA chains if needed
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not w3.isConnected():
            raise ConnectionError("Failed to connect to Ethereum node")
            
        logger.info(f"Connected to Ethereum node: {rpc_url}")
        return w3
        
    def _setup_account(self) -> Account:
        """Setup trading account from private key"""
        private_key = self.config.get('private_key')
        if not private_key:
            logger.warning("No private key provided - read-only mode")
            return None
            
        account = Account.from_key(private_key)
        logger.info(f"Trading account: {account.address}")
        return account
        
    def _load_contracts(self) -> Dict:
        """Load Uniswap contract interfaces"""
        contracts = {}
        
        # Load ABIs (simplified for example)
        router_abi = self._load_abi('uniswap_v3_router.json')
        factory_abi = self._load_abi('uniswap_v3_factory.json')
        quoter_abi = self._load_abi('uniswap_v3_quoter.json')
        pool_abi = self._load_abi('uniswap_v3_pool.json')
        
        # Create contract instances
        contracts['router'] = self.w3.eth.contract(
            address=UNISWAP_V3_ADDRESSES['router'],
            abi=router_abi
        )
        contracts['factory'] = self.w3.eth.contract(
            address=UNISWAP_V3_ADDRESSES['factory'],
            abi=factory_abi
        )
        contracts['quoter'] = self.w3.eth.contract(
            address=UNISWAP_V3_ADDRESSES['quoter'],
            abi=quoter_abi
        )
        
        return contracts
        
    def _load_abi(self, filename: str) -> List:
        """Load contract ABI from file"""
        # In production, load from actual ABI files
        # For now, return minimal ABI
        if filename == 'uniswap_v3_router.json':
            return [{
                "inputs": [{"type": "tuple", "name": "params"}],
                "name": "exactInputSingle",
                "outputs": [{"type": "uint256"}],
                "type": "function"
            }]
        return []
        
    async def get_pool_address(self, token0: str, token1: str, fee: int = 3000) -> str:
        """
        Get Uniswap V3 pool address for token pair
        
        Args:
            token0: First token address
            token1: Second token address  
            fee: Pool fee tier (500, 3000, 10000)
            
        Returns:
            Pool contract address
        """
        # Order tokens
        if int(token0, 16) > int(token1, 16):
            token0, token1 = token1, token0
            
        # Check cache
        cache_key = f"{token0}_{token1}_{fee}"
        if cache_key in self.pool_cache:
            return self.pool_cache[cache_key]
            
        # Get pool from factory
        pool_address = self.contracts['factory'].functions.getPool(
            Web3.toChecksumAddress(token0),
            Web3.toChecksumAddress(token1),
            fee
        ).call()
        
        if pool_address == '0x0000000000000000000000000000000000000000':
            raise ValueError(f"No pool found for {token0}/{token1} with fee {fee}")
            
        self.pool_cache[cache_key] = pool_address
        return pool_address
        
    async def get_quote(self, 
                       token_in: str,
                       token_out: str,
                       amount_in: int,
                       fee: int = 3000) -> Dict:
        """
        Get price quote for swap
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount (in token units with decimals)
            fee: Pool fee tier
            
        Returns:
            Quote information including output amount and price impact
        """
        try:
            # Use Quoter contract for accurate quotes
            amount_out = self.contracts['quoter'].functions.quoteExactInputSingle(
                Web3.toChecksumAddress(token_in),
                Web3.toChecksumAddress(token_out),
                fee,
                amount_in,
                0  # sqrtPriceLimitX96 (0 = no limit)
            ).call()
            
            # Calculate price impact
            pool_address = await self.get_pool_address(token_in, token_out, fee)
            pool_contract = self.w3.eth.contract(
                address=pool_address,
                abi=self._load_abi('uniswap_v3_pool.json')
            )
            
            # Get pool reserves for impact calculation
            slot0 = pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            
            # Calculate execution price and impact
            execution_price = amount_out / amount_in
            
            return {
                'amount_out': amount_out,
                'execution_price': execution_price,
                'price_impact': self._calculate_price_impact(
                    amount_in, amount_out, sqrt_price_x96
                ),
                'fee': fee,
                'pool': pool_address,
                'gas_estimate': 150000  # Typical gas for swap
            }
            
        except Exception as e:
            logger.error(f"Quote failed: {e}")
            return None
            
    async def execute_swap(self,
                          token_in: str,
                          token_out: str,
                          amount_in: int,
                          min_amount_out: Optional[int] = None,
                          fee: int = 3000) -> Optional[Dict]:
        """
        Execute token swap on Uniswap V3
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount
            min_amount_out: Minimum output (slippage protection)
            fee: Pool fee tier
            
        Returns:
            Transaction receipt or None if failed
        """
        if not self.account:
            logger.error("No account configured for trading")
            return None
            
        try:
            # Get quote first
            quote = await self.get_quote(token_in, token_out, amount_in, fee)
            if not quote:
                return None
                
            # Calculate minimum output with slippage
            if min_amount_out is None:
                min_amount_out = int(quote['amount_out'] * (1 - self.slippage_tolerance))
                
            # Build swap parameters
            deadline = int(datetime.now().timestamp()) + self.deadline_seconds
            
            swap_params = {
                'tokenIn': Web3.toChecksumAddress(token_in),
                'tokenOut': Web3.toChecksumAddress(token_out),
                'fee': fee,
                'recipient': self.account.address,
                'deadline': deadline,
                'amountIn': amount_in,
                'amountOutMinimum': min_amount_out,
                'sqrtPriceLimitX96': 0
            }
            
            # Build transaction
            swap_tx = self.contracts['router'].functions.exactInputSingle(
                swap_params
            ).buildTransaction({
                'from': self.account.address,
                'gas': quote['gas_estimate'],
                'gasPrice': await self._get_gas_price(),
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'value': amount_in if token_in == TOKENS['WETH'] else 0
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(swap_tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Swap transaction sent: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = await self._wait_for_transaction(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Swap successful: {tx_hash.hex()}")
                return {
                    'tx_hash': tx_hash.hex(),
                    'amount_in': amount_in,
                    'amount_out': min_amount_out,  # Actual amount from logs
                    'gas_used': receipt['gasUsed'],
                    'effective_gas_price': receipt['effectiveGasPrice']
                }
            else:
                logger.error(f"Swap failed: {tx_hash.hex()}")
                return None
                
        except Exception as e:
            logger.error(f"Swap execution failed: {e}")
            return None
            
    async def find_arbitrage_opportunities(self, 
                                         tokens: List[str],
                                         min_profit: float = 0.01) -> List[Dict]:
        """
        Find arbitrage opportunities across Uniswap pools
        
        Args:
            tokens: List of token addresses to check
            min_profit: Minimum profit percentage (0.01 = 1%)
            
        Returns:
            List of profitable arbitrage paths
        """
        opportunities = []
        
        # Check triangular arbitrage paths
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                for k in range(j + 1, len(tokens)):
                    path = [tokens[i], tokens[j], tokens[k], tokens[i]]
                    
                    opportunity = await self._check_arbitrage_path(path, min_profit)
                    if opportunity:
                        opportunities.append(opportunity)
                        
        # Sort by profit
        opportunities.sort(key=lambda x: x['profit_percent'], reverse=True)
        
        return opportunities
        
    async def _check_arbitrage_path(self, path: List[str], min_profit: float) -> Optional[Dict]:
        """Check if arbitrage path is profitable"""
        # Start with 1 ETH worth
        amount = int(1e18)
        current_amount = amount
        
        quotes = []
        total_gas = 0
        
        # Simulate swaps through path
        for i in range(len(path) - 1):
            token_in = path[i]
            token_out = path[i + 1]
            
            # Try different fee tiers
            best_quote = None
            for fee in [500, 3000, 10000]:
                try:
                    quote = await self.get_quote(token_in, token_out, current_amount, fee)
                    if quote and (not best_quote or quote['amount_out'] > best_quote['amount_out']):
                        best_quote = quote
                except:
                    continue
                    
            if not best_quote:
                return None
                
            quotes.append(best_quote)
            current_amount = best_quote['amount_out']
            total_gas += best_quote['gas_estimate']
            
        # Calculate profit
        profit = current_amount - amount
        profit_percent = profit / amount
        
        # Account for gas costs
        gas_cost_wei = total_gas * await self._get_gas_price()
        gas_cost_eth = gas_cost_wei / 1e18
        
        net_profit_eth = (profit / 1e18) - gas_cost_eth
        net_profit_percent = net_profit_eth
        
        if net_profit_percent >= min_profit:
            return {
                'path': path,
                'quotes': quotes,
                'profit_percent': profit_percent,
                'net_profit_percent': net_profit_percent,
                'net_profit_eth': net_profit_eth,
                'gas_cost_eth': gas_cost_eth,
                'total_gas': total_gas
            }
            
        return None
        
    async def provide_liquidity(self,
                               token0: str,
                               token1: str,
                               amount0: int,
                               amount1: int,
                               fee: int = 3000,
                               tick_lower: Optional[int] = None,
                               tick_upper: Optional[int] = None) -> Optional[Dict]:
        """
        Provide liquidity to Uniswap V3 pool
        
        Args:
            token0: First token address
            token1: Second token address
            amount0: Amount of token0
            amount1: Amount of token1
            fee: Pool fee tier
            tick_lower: Lower tick for concentrated liquidity
            tick_upper: Upper tick for concentrated liquidity
            
        Returns:
            Transaction receipt or None
        """
        # Implementation for liquidity provision
        # This would interact with the NonfungiblePositionManager contract
        pass
        
    async def monitor_pool_events(self, pool_address: str, callback: callable):
        """
        Monitor pool events in real-time
        
        Args:
            pool_address: Pool contract address
            callback: Function to call on each event
        """
        pool_contract = self.w3.eth.contract(
            address=pool_address,
            abi=self._load_abi('uniswap_v3_pool.json')
        )
        
        # Create event filter
        event_filter = pool_contract.events.Swap.createFilter(fromBlock='latest')
        
        while True:
            try:
                for event in event_filter.get_new_entries():
                    await callback(event)
                    
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                logger.error(f"Event monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def _get_gas_price(self) -> int:
        """Get current gas price with multiplier"""
        base_gas_price = self.w3.eth.gas_price
        
        # Apply multiplier for faster confirmation
        gas_price = int(base_gas_price * self.gas_price_multiplier)
        
        # Cap at maximum
        max_gas_wei = self.max_gas_price * 1e9
        gas_price = min(gas_price, max_gas_wei)
        
        return gas_price
        
    async def _wait_for_transaction(self, tx_hash: bytes, timeout: int = 300) -> Dict:
        """Wait for transaction confirmation"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    return receipt
            except:
                pass
                
            await asyncio.sleep(2)
            
        raise TimeoutError(f"Transaction {tx_hash.hex()} not confirmed in {timeout} seconds")
        
    def _calculate_price_impact(self, amount_in: int, amount_out: int, sqrt_price_x96: int) -> float:
        """Calculate price impact of swap"""
        # Simplified calculation - in production use proper math
        # Price = (sqrtPriceX96 / 2^96)^2
        price = (sqrt_price_x96 / (2**96)) ** 2
        
        execution_price = amount_out / amount_in
        spot_price = price  # This needs proper token decimal adjustment
        
        impact = abs(execution_price - spot_price) / spot_price
        return impact
        
    async def get_pool_stats(self, pool_address: str) -> Dict:
        """Get comprehensive pool statistics"""
        pool_contract = self.w3.eth.contract(
            address=pool_address,
            abi=self._load_abi('uniswap_v3_pool.json')
        )
        
        # Get pool data
        slot0 = pool_contract.functions.slot0().call()
        liquidity = pool_contract.functions.liquidity().call()
        
        # Get 24h volume from events
        volume_24h = await self._calculate_24h_volume(pool_address)
        
        # Calculate fees earned
        fees_24h = volume_24h * (pool_contract.functions.fee().call() / 1e6)
        
        return {
            'liquidity': liquidity,
            'sqrt_price_x96': slot0[0],
            'tick': slot0[1],
            'fee': pool_contract.functions.fee().call(),
            'volume_24h': volume_24h,
            'fees_24h': fees_24h,
            'tvl': self._calculate_tvl(pool_address, liquidity, slot0[0])
        }
        
    async def _calculate_24h_volume(self, pool_address: str) -> float:
        """Calculate 24-hour trading volume for pool"""
        # This would query historical events
        # Simplified for example
        return 1000000  # $1M volume
        
    def _calculate_tvl(self, pool_address: str, liquidity: int, sqrt_price_x96: int) -> float:
        """Calculate total value locked in pool"""
        # This requires proper calculation based on liquidity and price
        # Simplified for example
        return liquidity * 2  # Placeholder
