from aligned import ContractStore
import logging


async def custom_store() -> ContractStore:
    """
    Loads the contract store.
    This is needed for the cata catalog, as it is looking for this spesific function.
    Therefore, it will break if this is renamed.

    You can also manually spesify all the views, and models if that is prefered.
    """
    # Default is to read everything in the current dir and sub-dirs
    # If this starts to take too long, we can manually add the contracts we need
    logging.basicConfig(level=logging.INFO)
    return await load_store()


async def load_store() -> ContractStore:
    return await ContractStore.from_glob("src/**/contract*.py")
