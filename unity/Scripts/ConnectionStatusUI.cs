using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

/// <summary>
/// Manages the connection status UI
/// </summary>
public class ConnectionStatusUI : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI statusText;
    [SerializeField] private TextMeshProUGUI instructionsText;
    [SerializeField] private GameObject statusPanel;
    [SerializeField] private GameObject reconnectButton;
    
    [Header("Instructions")]
    [SerializeField] private string punchInstructions = "PUNCH: Move your arm forward quickly in a punching motion";
    [SerializeField] private string squatInstructions = "SQUAT: Bend your knees and lower your body";
    [SerializeField] private string plankInstructions = "PLANK: Hold your body in a straight line from head to toe";
    
    public void ShowStatus(string message, Color color)
    {
        if (statusPanel != null)
        {
            statusPanel.SetActive(true);
        }
        
        if (statusText != null)
        {
            statusText.text = message;
            statusText.color = color;
        }
    }
    
    public void ShowReconnectButton(bool show)
    {
        if (reconnectButton != null)
        {
            reconnectButton.SetActive(show);
        }
    }
    
    public void ShowInstructions(bool punch, bool squat, bool plank)
    {
        if (instructionsText == null) return;
        
        string instructions = "INSTRUCTIONS:\n\n";
        
        // Add instructions based on detected exercises
        if (punch)
        {
            instructions += "✓ " + punchInstructions + "\n\n";
        }
        else
        {
            instructions += "• " + punchInstructions + "\n\n";
        }
        
        if (squat)
        {
            instructions += "✓ " + squatInstructions + "\n\n";
        }
        else
        {
            instructions += "• " + squatInstructions + "\n\n";
        }
        
        if (plank)
        {
            instructions += "✓ " + plankInstructions;
        }
        else
        {
            instructions += "• " + plankInstructions;
        }
        
        instructionsText.text = instructions;
    }
    
    public void OnReconnectButtonClicked()
    {
        // Find the MotionDataReceiver in the scene
        MotionDataReceiver receiver = FindObjectOfType<MotionDataReceiver>();
        if (receiver != null)
        {
            // Reinitialize the connection
            receiver.gameObject.SetActive(false);
            receiver.gameObject.SetActive(true);
            
            ShowStatus("Reconnecting...", Color.yellow);
            ShowReconnectButton(false);
        }
    }
} 