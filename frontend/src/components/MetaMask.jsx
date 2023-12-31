import { useEffect } from 'react';
import { useState } from 'react';

export default function MetaMask() {

  const [currentAccount, setCurrentAccount] = useState(null);
  
  useEffect(() => {
    checkWalletIsConnected();
  }, [])

  const checkWalletIsConnected = async () => {
    const { ethereum } = window;

    if (!ethereum) {
      console.log("Make sure to have Metamask installed!");
      return;
    } else {
      console.log("Wallet exists! Happy trading!");
    }

    const accounts = await ethereum.request({ method: 'eth_accounts' });

    if (accounts.length !== 0) {
      const account = accounts[0];
      console.log("Found an authorized account: ", account);
      setCurrentAccount(account);
    } else {
      console.log("No authorized account found");
    }
  }
    
  const connectWalletHandler = async () => {
    const { ethereum } = window;

    if (!ethereum) {
      alert("Please install Metamask!");
    }

    try {
      const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
      console.log("Found an account! Address: ", accounts[0]);
      setCurrentAccount(accounts[0]);
    } catch (err) {
      console.log(err)
    }
  }
    
  const connectWalletButton = () => {
    return (
      <button onClick={connectWalletHandler} className='cta-button connect-wallet-button'>
        Connect Wallet
      </button>
    )
  }

  const walletConnectedButton = () => {
    return (
      <button className='cta-button mint-nft-button'>
        Wallet Connected!
      </button>
    )
  }

  return (
    <div>
        {currentAccount ? walletConnectedButton() : connectWalletButton()}
    </div>
  );
}
