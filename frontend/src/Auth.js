import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function Auth() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isRegister, setIsRegister] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");

    try {
      if (isRegister) {
        // Register User
        await axios.post("http://localhost:5001/register", {
          username,
          password,
        });
        setMessage("Registration successful! Please login.");
        setIsRegister(false);
      } else {
        // Login User
        const response = await axios.post("http://localhost:5001/login", {
          username,
          password,
        });

        const token = response.data.token;
        localStorage.setItem("token", token);
        localStorage.setItem("username", username);
        navigate("/app"); // 🔹 Redirect to prediction page
      }
    } catch (error) {
      setMessage(error.response?.data?.message || "Operation failed");
    }
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-2xl font-bold mb-4">{isRegister ? "Register" : "Login"}</h1>
      <form onSubmit={handleSubmit} className="w-80 bg-white p-6 rounded-lg shadow-md">
        <input
          type="text"
          placeholder="Username"
          className="w-full mb-3 p-2 border rounded"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full mb-3 p-2 border rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600">
          {isRegister ? "Register" : "Login"}
        </button>
      </form>
      {message && <p className="mt-4 text-red-500">{message}</p>}
      <button
        className="mt-3 text-sm text-blue-500"
        onClick={() => {
          setIsRegister(!isRegister);
          setMessage("");
        }}
      >
        {isRegister ? "Already have an account? Login" : "No account? Register"}
      </button>
    </div>
  );
}
