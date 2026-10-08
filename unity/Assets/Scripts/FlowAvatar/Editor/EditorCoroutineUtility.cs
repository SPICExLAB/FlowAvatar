using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Utility class for running Coroutines in the Unity Editor.
/// </summary>
public static class EditorCoroutineUtility
{
    private class EditorCoroutine
    {
        public IEnumerator Routine { get; private set; }
        public object Owner { get; private set; }
        public bool IsComplete { get; private set; }

        private readonly Stack<IEnumerator> _stack = new Stack<IEnumerator>();

        public EditorCoroutine(IEnumerator routine, object owner)
        {
            Routine = routine;
            Owner = owner;
            IsComplete = false;
            _stack.Push(routine);
        }

        public void MoveNext()
        {
            if (_stack.Count == 0)
            {
                IsComplete = true;
                return;
            }

            IEnumerator current = _stack.Peek();
            bool moveNext = current.MoveNext();

            if (!moveNext)
            {
                _stack.Pop();
                MoveNext();
                return;
            }

            object currentValue = current.Current;

            if (currentValue == null)
            {
                return;
            }

            if (currentValue is IEnumerator enumerator)
            {
                _stack.Push(enumerator);
                return;
            }

            if (currentValue is WaitForSeconds)
            {
                // Handle WaitForSeconds - in Editor we'll just wait for one update
                return;
            }

            if (currentValue is AsyncOperation asyncOp)
            {
                // Handle AsyncOperation - check if it's done each update
                if (!asyncOp.isDone)
                {
                    // Stay on this same step until the operation completes
                    _stack.Push(WaitForAsyncOperation(asyncOp));
                }
                return;
            }

            if (currentValue is CustomYieldInstruction customYield)
            {
                // Handle custom yield instructions
                if (customYield.keepWaiting)
                {
                    // Stay on this same step until keepWaiting is false
                    _stack.Push(WaitForCustomYield(customYield));
                }
                return;
            }
        }

        private IEnumerator WaitForAsyncOperation(AsyncOperation asyncOp)
        {
            while (!asyncOp.isDone)
            {
                yield return null;
            }
        }

        private IEnumerator WaitForCustomYield(CustomYieldInstruction customYield)
        {
            while (customYield.keepWaiting)
            {
                yield return null;
            }
        }
    }

    private static readonly List<EditorCoroutine> _activeCoroutines = new List<EditorCoroutine>();
    private static bool _initialized;

    /// <summary>
    /// Starts a coroutine in the Unity Editor.
    /// </summary>
    /// <param name="routine">The coroutine to start.</param>
    /// <param name="owner">The owner of the coroutine. Used for tracking and cancellation.</param>
    /// <returns>A reference to the started coroutine.</returns>
    public static object StartCoroutine(IEnumerator routine, object owner)
    {
        Initialize();

        EditorCoroutine coroutine = new EditorCoroutine(routine, owner);
        _activeCoroutines.Add(coroutine);
        return coroutine;
    }

    /// <summary>
    /// Stops all coroutines associated with the given owner.
    /// </summary>
    /// <param name="owner">The owner of the coroutines to stop.</param>
    public static void StopAllCoroutines(object owner)
    {
        _activeCoroutines.RemoveAll(c => c.Owner == owner);
    }

    private static void Initialize()
    {
        if (_initialized) return;

        EditorApplication.update += UpdateCoroutines;
        _initialized = true;
    }

    private static void UpdateCoroutines()
    {
        for (int i = _activeCoroutines.Count - 1; i >= 0; i--)
        {
            EditorCoroutine coroutine = _activeCoroutines[i];
            coroutine.MoveNext();

            if (coroutine.IsComplete)
            {
                _activeCoroutines.RemoveAt(i);
            }
        }
    }
}