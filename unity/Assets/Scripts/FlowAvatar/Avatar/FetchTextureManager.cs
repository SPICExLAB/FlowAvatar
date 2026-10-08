using System;
using System.Text.RegularExpressions;
using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using Oculus.Interaction; // For Meta's interaction system

/// <summary>
/// Manages virtual keyboard interactions with TextMeshPro input fields,
/// email validation, and camera rig tracking mode changes
/// </summary>
public class FetchTextureManager : OVRVirtualKeyboard.AbstractTextHandler
{
    [Header("Required References")]
    [SerializeField] private OVRVirtualKeyboard virtualKeyboard;
    [SerializeField] private TMP_InputField emailInputField;
    [SerializeField] private OVRManager cameraRig;
    [SerializeField] private AvatarFetcher avatarFetcher;
    [SerializeField] private GameObject Mirror;
    [SerializeField] private GameObject Floor;

    [Header("UI Elements")]
    [SerializeField] private GameObject UIs;
    [SerializeField] private GameObject submitButton;
    [SerializeField] private TextMeshProUGUI validationMessage;
    [SerializeField] private EmailValidationButtonVisual submitButtonVisual;

    [Header("Email Validation")]
    [SerializeField] private string invalidEmailMessage = "Please enter a valid email address";

    // AbstractTextHandler implementation
    public override string Text => emailInputField.text;
    public override bool SubmitOnEnter => true; // Always use enter as submit for email
    public override bool IsFocused => emailInputField.isFocused;

    private event Action<string> textChangedEvent;
    public override Action<string> OnTextChanged
    {
        get => textChangedEvent;
        set => textChangedEvent = value;
    }

    // Email validation regex pattern
    private static readonly Regex EmailRegex = new Regex(
        @"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        RegexOptions.Compiled);

    // State tracking
    private bool isInitialized = false;
    private InteractableUnityEventWrapper submitButtonEvents;

    private void Awake()
    {
        if (validationMessage != null)
        {
            validationMessage.gameObject.SetActive(false);
        }

        // Initially hide the keyboard
        if (virtualKeyboard != null)
        {
            virtualKeyboard.gameObject.SetActive(false);
        }
    }

    private void OnEnable()
    {
        if (!isInitialized)
        {
            SetupEventListeners();
        }
    }

    private void SetupEventListeners()
    {
        Debug.Log("Setting up event listeners");

        // Check for missing components
        if (virtualKeyboard == null)
        {
            Debug.LogError("No OVRVirtualKeyboard assigned!");
            return;
        }

        if (emailInputField == null)
        {
            Debug.LogError("No TMP_InputField assigned!");
            return;
        }

        // Set up input field events manually
        emailInputField.onSelect.AddListener(OnInputFieldSelect);
        emailInputField.onDeselect.AddListener(OnInputFieldDeselect);

        // Set up keyboard event listeners
        virtualKeyboard.TextHandler = this;
        virtualKeyboard.KeyboardShownEvent.AddListener(OnKeyboardShown);
        virtualKeyboard.KeyboardHiddenEvent.AddListener(OnKeyboardHidden);
        virtualKeyboard.EnterEvent.AddListener(OnKeyboardEnterPressed);

        // Subscribe to keyboard commit text event
        virtualKeyboard.CommitTextEvent.AddListener(OnKeyboardCommitText);
        virtualKeyboard.BackspaceEvent.AddListener(OnKeyboardBackspace);

        // Set up button state handlers (Meta's interaction system)
        if (submitButton != null)
        {
            submitButtonEvents = submitButton.GetComponent<InteractableUnityEventWrapper>();
            if (submitButtonEvents == null)
            {
                Debug.LogWarning("Submit button doesn't have InteractableUnityEventWrapper component!");
            }
            else
            {
                submitButtonEvents.WhenSelect.AddListener(OnSubmitButtonPressed);
            }
        }

        isInitialized = true;
        Debug.Log("Event listeners setup complete");
    }

    private void OnDestroy()
    {
        // Clean up event listeners
        if (emailInputField != null)
        {
            emailInputField.onSelect.RemoveListener(OnInputFieldSelect);
            emailInputField.onDeselect.RemoveListener(OnInputFieldDeselect);
        }

        if (virtualKeyboard != null)
        {
            virtualKeyboard.KeyboardShownEvent.RemoveListener(OnKeyboardShown);
            virtualKeyboard.KeyboardHiddenEvent.RemoveListener(OnKeyboardHidden);
            virtualKeyboard.EnterEvent.RemoveListener(OnKeyboardEnterPressed);
            virtualKeyboard.CommitTextEvent.RemoveListener(OnKeyboardCommitText);
            virtualKeyboard.BackspaceEvent.RemoveListener(OnKeyboardBackspace);

            if (virtualKeyboard.TextHandler == this)
            {
                virtualKeyboard.TextHandler = null;
            }
        }

        if (submitButtonEvents != null)
        {
            submitButtonEvents.WhenSelect.RemoveListener(OnSubmitButtonPressed);
        }
    }

    #region Event Handlers

    public void OnInputFieldSelect(string text)
    {
        Debug.Log("Input field selected");
        ShowKeyboard();
    }

    public void OnInputFieldDeselect(string text)
    {
        Debug.Log("Input field deselected");
        HideKeyboard();
    }

    private void OnKeyboardShown()
    {
        Debug.Log("Keyboard shown");
    }

    private void OnKeyboardHidden()
    {
        Debug.Log("Keyboard hidden");
    }

    private void OnKeyboardCommitText(string text)
    {
        Debug.Log($"Keyboard commit text: {text}");
        // Make sure to update the input field with the typed text
        AppendText(text);
    }

    private void OnKeyboardBackspace()
    {
        Debug.Log("Keyboard backspace");
        // Apply backspace to the input field
        ApplyBackspace();
    }

    private IEnumerator SetFloorLevelTrackingDelayed()
    {
        yield return null; // Wait one frame
        cameraRig.trackingOriginType = OVRManager.TrackingOrigin.FloorLevel;
    }

    private void OnKeyboardEnterPressed()
    {
        Debug.Log("Enter key pressed on virtual keyboard");
        // Process the submission
        ProcessSubmission();
    }

    private void OnSubmitButtonPressed()
    {
        Debug.Log("Submit button pressed");
        ProcessSubmission();
    }

    #endregion

    #region Public Methods
    public void ProcessSubmission()
    {
        // Validate email when submitting
        bool isEmailValid = ValidateEmail(emailInputField.text);

        if (isEmailValid)
        {
            Debug.Log($"Valid email submitted: {emailInputField.text}");

            // Submit the input
            Submit();

            // Hide keyboard and return to floor tracking
            HideKeyboard();

            UIs.SetActive(false);

            // Clear validation message if visible
            if (validationMessage != null)
            {
                validationMessage.gameObject.SetActive(false);
            }

            if (cameraRig != null)
            {
                // Use coroutine to wait a frame before changing tracking mode
                // This helps prevent visual artifacts during transition
                StartCoroutine(SetFloorLevelTrackingDelayed());
                Mirror.SetActive(true);
                Floor.SetActive(true);
            }

            avatarFetcher.FetchAvatarForUser(emailInputField.text);

        }
        else
        {
            ShowValidationError();
        }
    }
    public void ShowKeyboard()
    {
        virtualKeyboard.gameObject.SetActive(true);

        // Make sure the keyboard initially shows the current text
        if (virtualKeyboard != null && emailInputField != null)
        {
            virtualKeyboard.ChangeTextContext(emailInputField.text);
        }
    }

    public void HideKeyboard()
    {
        Debug.Log("Hiding keyboard");
        virtualKeyboard.gameObject.SetActive(false);
    }

    #endregion

    #region Private Helper Methods

    private bool ValidateEmail(string email)
    {
        return !string.IsNullOrEmpty(email) && EmailRegex.IsMatch(email);
    }

    private void ShowValidationError()
    {
        Debug.Log("asdasdasdasdasdasdasdasdasd");
        if (validationMessage != null)
        {
            validationMessage.text = invalidEmailMessage;
            validationMessage.gameObject.SetActive(true);

            // Show error indicator on the button
            if (submitButtonVisual != null)
            {
                // Store original select color state and temporarily override it
                StartCoroutine(FlashErrorColor());
            }

            // Hide the message after a delay
            StartCoroutine(HideValidationMessageAfterDelay(3f));
        }
    }

    private IEnumerator FlashErrorColor()
    {
        // Use our custom EmailValidationButtonVisual to show error state
        if (submitButtonVisual != null)
        {
            submitButtonVisual.ShowInvalidEmailState();
        }

        // This coroutine no longer needs to wait since ShowInvalidEmailState handles its own timing
        yield return null;
    }

    private IEnumerator HideValidationMessageAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        if (validationMessage != null)
        {
            validationMessage.gameObject.SetActive(false);
        }
    }

    

    #endregion

    #region AbstractTextHandler Implementation

    public override void Submit()
    {
        emailInputField.onEndEdit.Invoke(emailInputField.text);

        // Create and send a submit event
        var eventData = new BaseEventData(EventSystem.current);
        emailInputField.OnSubmit(eventData);
    }

    public override void AppendText(string text)
    {
        if (!emailInputField.isFocused)
        {
            return;
        }

        // Update text
        emailInputField.text += text;

        // Move caret to end
        emailInputField.caretPosition = emailInputField.text.Length;
        emailInputField.ForceLabelUpdate();

        // Make sure the visual update happens immediately
        emailInputField.SetTextWithoutNotify(emailInputField.text);

        // Notify about text change
        textChangedEvent?.Invoke(emailInputField.text);
    }

    public override void ApplyBackspace()
    {
        if (!emailInputField.isFocused || string.IsNullOrEmpty(emailInputField.text))
        {
            return;
        }

        // Remove last character
        emailInputField.text = emailInputField.text.Substring(0, emailInputField.text.Length - 1);

        // Move caret to end
        emailInputField.caretPosition = emailInputField.text.Length;
        emailInputField.ForceLabelUpdate();

        // Make sure the visual update happens immediately
        emailInputField.SetTextWithoutNotify(emailInputField.text);

        // Notify about text change
        textChangedEvent?.Invoke(emailInputField.text);
    }

    public override void MoveTextEnd()
    {
        if (emailInputField.isFocused)
        {
            emailInputField.caretPosition = emailInputField.text.Length;
        }
    }

    #endregion
}