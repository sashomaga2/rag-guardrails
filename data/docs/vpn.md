# VPN Access

## Client and portal
Northwind uses the GlobalProtect VPN client on all company laptops.
Connect to the portal address vpn.northwind.example when working outside the office.
The VPN client is installed automatically on new laptops through Company Portal.

## Multi-factor authentication
Every VPN login requires multi-factor authentication with the Microsoft Authenticator app.
SMS codes are not accepted for VPN logins since March 2025.
If you get a new phone, re-register Authenticator at id.northwind.example before you wipe the old phone.

## Split tunneling
Split tunneling is disabled, so all traffic goes through the corporate network while the VPN is connected.
Personal streaming services may be slow on VPN; this is expected.

## Troubleshooting
If the VPN disconnects every few minutes, update the GlobalProtect client to the latest version from Company Portal.
If the error "Gateway not reachable" appears, check that your local Wi-Fi works and then reconnect to vpn.northwind.example.
For persistent issues contact the IT helpdesk at helpdesk@northwind.example or call extension 4357.
