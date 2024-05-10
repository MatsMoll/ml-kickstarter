from aligned import ContractStore



async def custom_store() -> ContractStore | None:
    # Default is to read everything in the current dir and sub-dirs
    return await ContractStore.from_dir()
