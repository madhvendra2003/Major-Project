//SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract SourceReputation {
    address public owner;
    mapping(string => uint) public trustScores;

    constructor() {
        owner = msg.sender;
    }

    function setTrust(string memory sourceId, uint trust) public {
        require(msg.sender == owner, "Only owner can set trust");
        trustScores[sourceId] = trust;
    }

    function getTrust(string memory sourceId) public view returns (uint) {
        return trustScores[sourceId];
    }
}