"""URIResolver for ARCO mirror with .netrc authentication."""

import netrc

from hydromt.data_catalog.uri_resolvers.uri_resolver import URIResolver


class ArcoMirrorResolver(URIResolver):
    """URIResolver for ARCO mirror with .netrc authentication."""

    name = "ARCO_mirror"

    def resolve(self, uri: str, **kwargs) -> list[str]:
        """Resolve the ARCO mirror URI, potentially using .netrc for credentials."""
        # Check if credentials are needed
        if "{" in uri:
            try:
                credentials = netrc.netrc()
                auth = credentials.authenticators(self.options.get("authenticator"))
                if auth is not None:
                    _, _, password = auth
                    uri = uri.format(password=password)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"No .netrc credentials file found for {uri}"
                ) from e

        return [uri]
