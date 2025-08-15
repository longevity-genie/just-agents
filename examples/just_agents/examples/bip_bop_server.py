from fastmcp import FastMCP

"""
Bip-Bop Server 🎵

 📊 Available Tools:
• bip() - Returns "BIP"
• bop() - Returns "BOP"
"""

mcp = FastMCP("Bip-Bop Server 🎵")

@mcp.tool()
def bip() -> str:
    """Make a bip sound"""
    return "BIP"

@mcp.tool()
def bop() -> str:
    """Make a bop sound"""
    return "BOP"

if __name__ == "__main__":
    mcp.run()
