from aligned import ContractStore



async def custom_store() -> ContractStore | None:
    # Default is to read everything in the current dir and sub-dirs
    # If this starts to take too long, we can manually add the contracts we need
    return await ContractStore.from_dir()
