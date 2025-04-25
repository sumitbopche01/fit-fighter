using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Controls a game object based on motion detection data
/// </summary>
public class MotionControlledObject : MonoBehaviour
{
    [Header("Motion Receiver")]
    [SerializeField] private MotionDataReceiver motionDataReceiver;
    
    [Header("Movement Settings")]
    [SerializeField] private float punchForce = 5f;
    [SerializeField] private float squatHeight = 2f;
    [SerializeField] private float returnSpeed = 3f;
    [SerializeField] private float plankStabilizationFactor = 0.5f;
    
    [Header("Debug")]
    [SerializeField] private TMPro.TextMeshProUGUI debugText;
    
    // Original position for reset
    private Vector3 originalPosition;
    private Vector3 targetPosition;
    private Rigidbody rb;
    
    // Exercise states
    private bool isPunching = false;
    private bool isSquatting = false;
    private bool isPlanking = false;
    
    private void Start()
    {
        // Store original position
        originalPosition = transform.position;
        targetPosition = originalPosition;
        
        // Get or add rigidbody
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
            rb.constraints = RigidbodyConstraints.FreezeRotation;
        }
        
        // Subscribe to motion data events
        if (motionDataReceiver != null)
        {
            motionDataReceiver.OnMotionDataReceived += OnMotionDataReceived;
        }
        else
        {
            Debug.LogError("MotionDataReceiver reference not set!");
        }
    }
    
    private void OnMotionDataReceived(MotionData data)
    {
        // Update exercise states
        isPunching = data.exercises.punch;
        isSquatting = data.exercises.squat;
        isPlanking = data.exercises.plank;
        
        // Update debug text
        UpdateDebugText(data);
    }
    
    private void Update()
    {
        // Apply motion-based actions
        HandlePunchMovement();
        HandleSquatMovement();
        HandlePlankStabilization();
        
        // Update motion target
        MoveTowardsTarget();
    }
    
    private void HandlePunchMovement()
    {
        if (isPunching)
        {
            // Move forward when punching
            targetPosition = originalPosition + transform.forward * punchForce;
            
            // Add immediate force
            rb.AddForce(transform.forward * punchForce, ForceMode.Impulse);
            
            // Visual feedback
            GetComponent<Renderer>().material.color = Color.red;
        }
    }
    
    private void HandleSquatMovement()
    {
        if (isSquatting)
        {
            // Move down when squatting
            targetPosition = new Vector3(
                originalPosition.x,
                originalPosition.y - squatHeight,
                originalPosition.z
            );
            
            // Visual feedback
            GetComponent<Renderer>().material.color = Color.blue;
        }
    }
    
    private void HandlePlankStabilization()
    {
        if (isPlanking)
        {
            // Hold position steady when planking
            rb.velocity *= plankStabilizationFactor;
            rb.angularVelocity *= plankStabilizationFactor;
            
            // Visual feedback
            GetComponent<Renderer>().material.color = Color.green;
        }
        else if (!isPunching && !isSquatting)
        {
            // Return to original position when no exercise is detected
            targetPosition = originalPosition;
            
            // Reset color
            GetComponent<Renderer>().material.color = Color.white;
        }
    }
    
    private void MoveTowardsTarget()
    {
        // Smoothly move towards target position
        transform.position = Vector3.Lerp(
            transform.position,
            targetPosition,
            returnSpeed * Time.deltaTime
        );
    }
    
    private void UpdateDebugText(MotionData data)
    {
        if (debugText != null)
        {
            string status = "Exercises: ";
            status += data.exercises.punch ? "PUNCH " : "";
            status += data.exercises.squat ? "SQUAT " : "";
            status += data.exercises.plank ? "PLANK " : "";
            
            if (!data.exercises.punch && !data.exercises.squat && !data.exercises.plank)
            {
                status += "None";
            }
            
            debugText.text = status;
        }
    }
    
    private void OnDestroy()
    {
        // Unsubscribe from events
        if (motionDataReceiver != null)
        {
            motionDataReceiver.OnMotionDataReceived -= OnMotionDataReceived;
        }
    }
} 